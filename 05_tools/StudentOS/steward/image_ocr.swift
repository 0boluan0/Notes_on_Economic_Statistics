import Foundation
import Vision

// Text extraction only. Completion decisions still require task-specific evidence.
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["en-US", "zh-Hans"]
request.usesLanguageCorrection = false
do {
    try VNImageRequestHandler(url: url).perform([request])
    for observation in request.results ?? [] {
        if let candidate = observation.topCandidates(1).first, candidate.confidence >= 0.7 {
            print(candidate.string)
        }
    }
} catch {
    fputs("Image text extraction failed\n", stderr)
    exit(1)
}
