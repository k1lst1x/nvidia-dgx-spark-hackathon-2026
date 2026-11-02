import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = VisionMateViewModel()

    var body: some View {
        ZStack {
            CameraPreview(session: viewModel.camera.session)
                .ignoresSafeArea()
                .accessibilityHidden(true)

            LinearGradient(
                colors: [.black.opacity(0.75), .clear, .black.opacity(0.88)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 18) {
                header
                Spacer(minLength: 8)
                answerCard
                actionButtons
                voiceButton
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
        }
        .preferredColorScheme(.dark)
        .task { viewModel.prepare() }
        .onDisappear { viewModel.camera.stop() }
        .sheet(isPresented: $viewModel.showSettings) {
            SettingsView(viewModel: viewModel)
        }
        .alert("VisionMate", isPresented: Binding(
            get: { viewModel.errorMessage != nil },
            set: { if !$0 { viewModel.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { viewModel.errorMessage = nil }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 3) {
                Text("VisionMate")
                    .font(.largeTitle.bold())
                Text(viewModel.status)
                    .font(.headline)
                    .foregroundStyle(.white.opacity(0.84))
            }
            Spacer()
            Button {
                viewModel.showSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .font(.title2)
                    .frame(width: 54, height: 54)
                    .background(.white.opacity(0.16), in: Circle())
            }
            .accessibilityLabel("Server settings")
            .accessibilityHint("Set the DGX Spark server address, model, and API key")
        }
    }

    private var answerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: viewModel.isAnalyzing ? "sparkles" : "speaker.wave.2.fill")
                Text(viewModel.isAnalyzing ? "Thinking" : "VisionMate says")
                    .font(.headline)
            }
            .foregroundStyle(.mint)

            Text(viewModel.isAnalyzing ? "One moment…" : viewModel.answer)
                .font(.title3.weight(.medium))
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityLabel("VisionMate response: \(viewModel.answer)")

            Text("AI can be wrong. Do not use it alone for traffic, medication, machinery, or emergencies.")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.yellow.opacity(0.9))

            Button {
                viewModel.toggleAnswerSpeech()
            } label: {
                Label(
                    viewModel.speechOutput.isSpeaking ? "Stop speaking" : "Repeat answer",
                    systemImage: viewModel.speechOutput.isSpeaking ? "stop.circle.fill" : "speaker.wave.2.circle.fill"
                )
                .font(.headline)
            }
            .buttonStyle(.bordered)
            .tint(.mint)
            .accessibilityHint(viewModel.speechOutput.isSpeaking ? "Stops the current spoken answer" : "Reads the current answer aloud")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(.white.opacity(0.22), lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isStaticText)
    }

    private var actionButtons: some View {
        HStack(spacing: 12) {
            ForEach([VisionMode.scene, .read]) { mode in
                Button {
                    viewModel.analyze(mode)
                } label: {
                    VStack(spacing: 9) {
                        Image(systemName: mode.icon)
                            .font(.title2)
                        Text(mode.title)
                            .font(.headline)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, minHeight: 96)
                }
                .buttonStyle(ActionCardStyle())
                .disabled(viewModel.isAnalyzing || !viewModel.camera.isConfigured)
                .accessibilityHint(mode == .scene ? "Takes a photo and describes it aloud" : "Takes a photo and reads visible text aloud")
            }
        }
    }

    private var voiceButton: some View {
        Button {
            viewModel.toggleVoiceQuestion()
        } label: {
            HStack(spacing: 12) {
                Image(systemName: viewModel.speechInput.isListening ? "stop.fill" : "mic.fill")
                    .font(.title2)
                Text(viewModel.speechInput.isListening ? "Stop and ask" : "Ask about what’s visible")
                    .font(.title3.bold())
                Spacer()
                if viewModel.speechInput.isListening {
                    Image(systemName: "waveform")
                        .symbolEffect(.variableColor.iterative)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 66)
            .padding(.horizontal, 20)
        }
        .buttonStyle(VoiceButtonStyle(isListening: viewModel.speechInput.isListening))
        .disabled(viewModel.isAnalyzing || !viewModel.camera.isConfigured)
        .accessibilityHint(viewModel.speechInput.isListening
            ? "Stops recording, takes a photo, and asks the question you just said"
            : "Starts recording a question about the camera view")
        .accessibilityValue(viewModel.speechInput.isListening ? "Listening. \(viewModel.speechInput.transcript)" : "Not listening")
    }
}

private struct ActionCardStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(.white)
            .background(configuration.isPressed ? .blue.opacity(0.55) : .blue.opacity(0.78), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
    }
}

private struct VoiceButtonStyle: ButtonStyle {
    let isListening: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(.black)
            .background(isListening ? .orange : .mint, in: Capsule())
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
    }
}

private struct SettingsView: View {
    @ObservedObject var viewModel: VisionMateViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var endpoint = ""
    @State private var model = ""
    @State private var apiKey = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("DGX Spark server") {
                    TextField("Server URL", text: $endpoint)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    TextField("Vision model", text: $model)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("API key (optional for local server)", text: $apiKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("Connection format") {
                    Text("VisionMate sends a single JPEG to an OpenAI-compatible /v1/chat/completions endpoint. Use a vision-language model, such as Qwen3-VL-32B-Instruct, rather than text-only Qwen3-32B.")
                }

                Section("Privacy") {
                    Text("Your camera image and spoken question are sent only to the server address above. The API key is stored in the iPhone Keychain.")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        viewModel.saveConfiguration(endpoint: endpoint, model: model, apiKey: apiKey)
                    }
                }
            }
        }
        .onAppear {
            endpoint = viewModel.configuration.endpoint
            model = viewModel.configuration.model
            apiKey = viewModel.apiKey
        }
    }
}

#Preview {
    ContentView()
}
