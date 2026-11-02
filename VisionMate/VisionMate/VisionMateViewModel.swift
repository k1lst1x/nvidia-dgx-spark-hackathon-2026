import Foundation

@MainActor
final class VisionMateViewModel: ObservableObject {
    @Published private(set) var answer = "Point your camera, then choose Describe scene, Read text, or Ask a question."
    @Published private(set) var status = "Preparing camera"
    @Published private(set) var isAnalyzing = false
    @Published var configuration: VisionConfiguration
    @Published var apiKey: String
    @Published var showSettings = false
    @Published var errorMessage: String?

    let camera = CameraService()
    let speechInput = SpeechInputService()
    let speechOutput = SpeechOutputService()

    private let api = VisionAPI()
    private static let configurationKey = "visionmate.configuration"

    init() {
        configuration = Self.loadConfiguration()
        apiKey = SecureStore.loadAPIKey()
    }

    func prepare() {
        Task {
            await camera.prepare()
            status = camera.isAuthorized ? "Ready" : "Camera permission needed"
        }
    }

    func analyze(_ mode: VisionMode, question: String? = nil) {
        guard !isAnalyzing else { return }
        speechOutput.stop()
        Task {
            isAnalyzing = true
            errorMessage = nil
            status = "Taking photo"
            do {
                let photo = try await camera.capturePhoto()
                status = "Asking VisionMate"
                let response = try await api.analyze(
                    imageData: photo,
                    mode: mode,
                    userQuestion: question,
                    configuration: configuration,
                    apiKey: apiKey
                )
                answer = response
                status = "Answer ready"
                speechOutput.speak(response)
            } catch {
                let message = error.localizedDescription
                answer = message
                errorMessage = message
                status = "Could not complete request"
                speechOutput.speak(message)
            }
            isAnalyzing = false
        }
    }

    func toggleVoiceQuestion() {
        if speechInput.isListening {
            speechInput.stop()
            let question = speechInput.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            analyze(.question, question: question.isEmpty ? nil : question)
            return
        }

        Task {
            guard await speechInput.requestAuthorization() else {
                let message = "Speech recognition permission is needed to ask a question by voice."
                answer = message
                errorMessage = message
                return
            }
            do {
                speechOutput.stop()
                try speechInput.start()
                status = "Listening. Tap Stop and ask when you finish."
            } catch {
                errorMessage = error.localizedDescription
                answer = error.localizedDescription
            }
        }
    }

    func toggleAnswerSpeech() {
        if speechOutput.isSpeaking {
            speechOutput.stop()
        } else {
            speechOutput.speak(answer)
        }
    }

    func saveConfiguration(endpoint: String, model: String, apiKey: String) {
        configuration = VisionConfiguration(
            endpoint: endpoint.trimmingCharacters(in: .whitespacesAndNewlines),
            model: model.trimmingCharacters(in: .whitespacesAndNewlines)
        )
        self.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = try? JSONEncoder().encode(configuration) {
            UserDefaults.standard.set(data, forKey: Self.configurationKey)
        }
        do {
            try SecureStore.saveAPIKey(self.apiKey)
            showSettings = false
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private static func loadConfiguration() -> VisionConfiguration {
        guard let data = UserDefaults.standard.data(forKey: configurationKey),
              let saved = try? JSONDecoder().decode(VisionConfiguration.self, from: data) else {
            return .default
        }
        return saved
    }
}
