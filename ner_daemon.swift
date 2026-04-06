import NaturalLanguage
import Foundation

// [P3-1] Long-running NER daemon — reads lines from stdin, writes results to stdout
// Protocol: one JSON line per request, one JSON line per response
// Eliminates ~50-80ms subprocess spawn overhead per transcription

let tagger = NLTagger(tagSchemes: [.nameType, .lexicalClass])
let recognizer = NLLanguageRecognizer()

func isCleanTerm(_ word: String) -> Bool {
    if word.count < 2 { return false }
    if word.contains("\u{FFFD}") || word.unicodeScalars.contains(where: { $0.value < 32 }) {
        return false
    }
    if !word.contains(where: { $0.isLetter }) { return false }

    // [BUG-7] Fixed ASCII letter range (was 65-122 which includes [\]^_`)
    let nonAscii = word.unicodeScalars.filter { !$0.isASCII }
    let asciiLetters = word.unicodeScalars.filter {
        ($0.value >= 65 && $0.value <= 90) || ($0.value >= 97 && $0.value <= 122)
    }
    if nonAscii.count > 0 && asciiLetters.count > 0 {
        return false
    }

    // Reject embedded digits: letter-digit-letter pattern (OCR substitution like Ba8h)
    let chars = Array(word)
    for i in 1..<(chars.count - 1) {
        if chars[i].isNumber && chars[i-1].isLetter && chars[i+1].isLetter {
            return false
        }
    }

    // Reject scattered digits in otherwise alphabetic terms
    let digitPositions = chars.indices.filter { chars[$0].isNumber }
    if digitPositions.count > 1 {
        let contiguous = zip(digitPositions, digitPositions.dropFirst()).allSatisfy { $1 == $0 + 1 }
        if !contiguous { return false }
    }

    // Reject random case zigzag (base64-like: AAQAT9bPvJQ)
    let lettersOnly = chars.filter { $0.isLetter }
    if lettersOnly.count >= 6 {
        var zigzags = 0
        for i in 2..<lettersOnly.count {
            if lettersOnly[i].isUppercase != lettersOnly[i-1].isUppercase &&
               lettersOnly[i-1].isUppercase != lettersOnly[i-2].isUppercase {
                zigzags += 1
            }
        }
        if zigzags >= 3 { return false }
    }

    // [BUG-7] Allow short alphanumeric terms (GPT4, M1, H100)
    let digits = word.filter { $0.isNumber }
    let letters = word.filter { $0.isLetter }
    if digits.count > 0 && letters.count > 0 && word.count > 4 {
        let ratio = Double(digits.count) / Double(word.count)
        if ratio > 0.3 && ratio < 0.7 { return false }
    }
    // [BUG-7] Raised threshold: allow ANTHROPIC, MICROSOFT (was >8)
    if word.count > 15 && word == word.uppercased() { return false }
    return true
}

func extractEntities(_ input: String) -> [String] {
    tagger.string = input
    let fullRange = input.startIndex..<input.endIndex

    recognizer.reset()
    recognizer.processString(input)
    if let lang = recognizer.dominantLanguage {
        tagger.setLanguage(lang, range: fullRange)
    }

    var seen = Set<String>()
    var results: [String] = []

    tagger.enumerateTags(in: fullRange, unit: .word, scheme: .nameType,
        options: [.omitPunctuation, .omitWhitespace, .joinNames]) { tag, range in
        if let tag = tag, tag != .otherWord {
            let word = String(input[range])
            let key = word.lowercased()
            if !seen.contains(key) && isCleanTerm(word) {
                seen.insert(key)
                results.append(word)
            }
        }
        return true
    }

    tagger.enumerateTags(in: fullRange, unit: .word, scheme: .lexicalClass,
        options: [.omitPunctuation, .omitWhitespace]) { tag, range in
        if let tag = tag, tag == .noun {
            let word = String(input[range])
            let isCapitalized = word.first?.isUppercase == true
            let hasMixedCase = word.contains(where: { $0.isUppercase }) &&
                              word.contains(where: { $0.isLowercase }) &&
                              word.dropFirst().contains(where: { $0.isUppercase })
            let hasDigits = word.contains(where: { $0.isNumber })
            if (isCapitalized || hasMixedCase || hasDigits) && word.count > 1 {
                let key = word.lowercased()
                if !seen.contains(key) && isCleanTerm(word) {
                    seen.insert(key)
                    results.append(word)
                }
            }
        }
        return true
    }

    return results
}

// Daemon loop: read lines from stdin, process, write results
// Signal readiness
fputs("READY\n", stdout)
fflush(stdout)

while let line = readLine(strippingNewline: true) {
    if line.isEmpty { continue }
    let entities = extractEntities(line)
    // Output: newline-separated terms, terminated by an empty line
    for entity in entities {
        print(entity)
    }
    print("")  // empty line = end of results for this request
    fflush(stdout)
}
