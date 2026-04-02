import NaturalLanguage
import Foundation

// Read text from stdin
let data = FileHandle.standardInput.readDataToEndOfFile()
guard let input = String(data: data, encoding: .utf8), !input.isEmpty else {
    print("")
    exit(0)
}

let tagger = NLTagger(tagSchemes: [.nameType, .lexicalClass])
tagger.string = input

let fullRange = input.startIndex..<input.endIndex

// Detect language and set hint
let recognizer = NLLanguageRecognizer()
recognizer.processString(input)
if let lang = recognizer.dominantLanguage {
    tagger.setLanguage(lang, range: fullRange)
}

var seen = Set<String>()
var results: [String] = []

// [P2-6] Filter: reject OCR garbage
func isCleanTerm(_ word: String) -> Bool {
    // Reject single characters
    if word.count < 2 { return false }

    // Reject terms with Unicode replacement chars or control chars
    if word.contains("\u{FFFD}") || word.unicodeScalars.contains(where: { $0.value < 32 }) {
        return false
    }
    // Reject terms that are entirely non-letter (e.g., "￿.￿", "$+<>")
    if !word.contains(where: { $0.isLetter }) { return false }

    // Reject terms with non-ASCII artifacts (OCR garbage like "Él", "ÈThllfeature", "￿")
    let asciiLetters = word.unicodeScalars.filter { $0.isASCII && ($0.value >= 65 && $0.value <= 122) }
    let nonAscii = word.unicodeScalars.filter { !$0.isASCII }
    if nonAscii.count > 0 && asciiLetters.count > 0 {
        // Mixed ASCII + non-ASCII like "CIaUdeÈ" — likely OCR garbage
        return false
    }

    // Reject terms that look like garbled OCR: random mixed digits+letters
    // e.g., "8y8te", "1tsf1", "a550ciated", "IJ17AfjL"
    let digits = word.filter { $0.isNumber }
    let letters = word.filter { $0.isLetter }
    if digits.count > 0 && letters.count > 0 {
        let ratio = Double(digits.count) / Double(word.count)
        // If 20-80% digits, likely garbage (pure numbers or pure text are fine)
        if ratio > 0.2 && ratio < 0.8 { return false }
    }

    // Reject very short ALL-CAPS that are common OCR noise
    // (real acronyms are usually 2-5 chars)
    if word.count > 8 && word == word.uppercased() { return false }

    return true
}

// 1. Named entities (PersonalName, OrganizationName, PlaceName)
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

// 2. Proper nouns via POS tagging (catches technical terms NER misses)
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

print(results.joined(separator: "\n"))
