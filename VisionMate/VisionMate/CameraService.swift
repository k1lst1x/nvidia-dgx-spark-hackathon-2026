import AVFoundation
import SwiftUI

@MainActor
final class CameraService: NSObject, ObservableObject {
    let session = AVCaptureSession()
    @Published private(set) var isAuthorized = false
    @Published private(set) var isConfigured = false
    @Published private(set) var errorMessage: String?

    private let output = AVCapturePhotoOutput()
    private var captureContinuation: CheckedContinuation<Data, Error>?

    func prepare() async {
        let granted = await requestCameraAccess()
        isAuthorized = granted
        guard granted else {
            errorMessage = "Camera access is required to describe what is in front of you."
            return
        }
        configureIfNeeded()
    }

    func start() {
        guard isConfigured, !session.isRunning else { return }
        session.startRunning()
    }

    func stop() {
        guard session.isRunning else { return }
        session.stopRunning()
    }

    func capturePhoto() async throws -> Data {
        guard isConfigured else { throw CameraError.notReady }
        return try await withCheckedThrowingContinuation { continuation in
            guard captureContinuation == nil else {
                continuation.resume(throwing: CameraError.captureInProgress)
                return
            }
            captureContinuation = continuation
            let settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
            settings.photoQualityPrioritization = .speed
            output.capturePhoto(with: settings, delegate: self)
        }
    }

    private func requestCameraAccess() async -> Bool {
        if AVCaptureDevice.authorizationStatus(for: .video) == .authorized { return true }
        return await AVCaptureDevice.requestAccess(for: .video)
    }

    private func configureIfNeeded() {
        guard !isConfigured else { return }
        session.beginConfiguration()
        session.sessionPreset = .photo
        defer { session.commitConfiguration() }

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            errorMessage = "The rear camera is not available on this device."
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            guard session.canAddInput(input), session.canAddOutput(output) else {
                errorMessage = "VisionMate could not prepare the camera."
                return
            }
            session.addInput(input)
            session.addOutput(output)
            isConfigured = true
            start()
        } catch {
            errorMessage = "VisionMate could not access the camera: \(error.localizedDescription)"
        }
    }
}

extension CameraService: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let result: Result<Data, Error>
        if let error {
            result = .failure(error)
        } else if let imageData = photo.fileDataRepresentation() {
            result = .success(imageData)
        } else {
            result = .failure(CameraError.noImageData)
        }

        Task { @MainActor [weak self] in
            guard let continuation = self?.captureContinuation else { return }
            self?.captureContinuation = nil
            continuation.resume(with: result)
        }
    }
}

enum CameraError: LocalizedError {
    case notReady
    case captureInProgress
    case noImageData

    var errorDescription: String? {
        switch self {
        case .notReady: "The camera is not ready yet."
        case .captureInProgress: "VisionMate is already taking a photo."
        case .noImageData: "VisionMate could not capture an image."
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}
