import Foundation

enum VisionMode: String, CaseIterable, Identifiable {
    case scene
    case read
    case question

    var id: String { rawValue }

    var title: String {
        switch self {
        case .scene: "Describe scene"
        case .read: "Read text"
        case .question: "Ask a question"
        }
    }

    var icon: String {
        switch self {
        case .scene: "eye.fill"
        case .read: "text.viewfinder"
        case .question: "mic.fill"
        }
    }

    var instruction: String {
        switch self {
        case .scene:
            "Describe the important objects in front of the user and their approximate locations. Start with the most useful object."
        case .read:
            "Read all clearly legible text exactly as written. State when text is too small, blurry, blocked, or uncertain instead of guessing."
        case .question:
            "Answer the user's question about the image. If the image does not support a confident answer, say what is unclear."
        }
    }
}

struct VisionConfiguration: Codable, Equatable {
    var endpoint: String
    var model: String

    static let `default` = VisionConfiguration(
        endpoint: "http://dgx-spark.local:8000/v1",
        model: "Qwen/Qwen3-VL-32B-Instruct"
    )
}

enum VisionAPIError: LocalizedError {
    case invalidEndpoint
    case invalidResponse
    case server(String)

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            "The DGX Spark server address is not valid. Open Settings and check it."
        case .invalidResponse:
            "The vision server returned a response VisionMate could not understand."
        case .server(let message):
            message
        }
    }
}

struct VisionAPI {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func analyze(
        imageData: Data,
        mode: VisionMode,
        userQuestion: String?,
        configuration: VisionConfiguration,
        apiKey: String
    ) async throws -> String {
        let endpoint = normalizedEndpoint(configuration.endpoint)
        guard let url = URL(string: endpoint) else { throw VisionAPIError.invalidEndpoint }

        let dataURL = "data:image/jpeg;base64,\(imageData.base64EncodedString())"
        let prompt = mode.instruction + "\n\n" + (userQuestion?.isEmpty == false
            ? "User question: \(userQuestion!)"
            : "There is no additional question.")
        let requestBody = ChatRequest(
            model: configuration.model,
            messages: [
                .init(role: "system", content: .text(systemPrompt)),
                .init(role: "user", content: .parts([
                    .text(prompt),
                    .image(URLImage(url: dataURL, detail: "low"))
                ]))
            ],
            temperature: 0.2,
            maxTokens: mode == .read ? 350 : 160
        )

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 45
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONEncoder().encode(requestBody)

        let (responseData, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw VisionAPIError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let details = (try? JSONDecoder().decode(ServerError.self, from: responseData))?.error.message
                ?? String(data: responseData, encoding: .utf8)
                ?? "Unknown server error"
            throw VisionAPIError.server("The vision server responded with \(httpResponse.statusCode): \(details)")
        }

        let completion = try JSONDecoder().decode(ChatCompletion.self, from: responseData)
        guard let answer = completion.choices.first?.message.content?.trimmingCharacters(in: .whitespacesAndNewlines),
              !answer.isEmpty else {
            throw VisionAPIError.invalidResponse
        }
        return answer
    }

    private func normalizedEndpoint(_ endpoint: String) -> String {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if trimmed.hasSuffix("/v1/chat/completions") { return trimmed }
        if trimmed.hasSuffix("/v1") { return trimmed + "/chat/completions" }
        return trimmed + "/v1/chat/completions"
    }

    private let systemPrompt = """
    You are VisionMate, a careful voice assistant for blind and low-vision people. Give a direct, useful answer in plain language that is comfortable to hear aloud. Mention positions using left, centre, right, or clock-face locations when helpful. Do not claim certainty when the image is blurry, blocked, or ambiguous. Never advise that it is safe to cross a road, take medication, operate machinery, or respond to an emergency from this image alone. Keep answers to two short sentences unless asked to read text.
    """
}

private struct ChatRequest: Encodable {
    let model: String
    let messages: [ChatMessage]
    let temperature: Double
    let maxTokens: Int

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature
        case maxTokens = "max_tokens"
    }
}

private struct ChatMessage: Encodable {
    let role: String
    let content: MessageContent
}

private enum MessageContent: Encodable {
    case text(String)
    case parts([ContentPart])

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let value): try container.encode(value)
        case .parts(let parts): try container.encode(parts)
        }
    }
}

private enum ContentPart: Encodable {
    case text(String)
    case image(URLImage)

    enum CodingKeys: String, CodingKey { case type, text, imageURL = "image_url" }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let value):
            try container.encode("text", forKey: .type)
            try container.encode(value, forKey: .text)
        case .image(let value):
            try container.encode("image_url", forKey: .type)
            try container.encode(value, forKey: .imageURL)
        }
    }
}

private struct URLImage: Encodable {
    let url: String
    let detail: String
}

private struct ChatCompletion: Decodable {
    let choices: [Choice]
    struct Choice: Decodable {
        let message: Message
    }
    struct Message: Decodable {
        let content: String?
    }
}

private struct ServerError: Decodable {
    let error: ErrorBody
    struct ErrorBody: Decodable { let message: String }
}
