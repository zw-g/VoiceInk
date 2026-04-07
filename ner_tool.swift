// One-shot NER tool — reads text from stdin, prints entities, exits
// Shared functions (isCleanTerm, extractEntities) live in ner_common.swift

import Foundation

@main
struct NERTool {
    static func main() {
        let data = FileHandle.standardInput.readDataToEndOfFile()
        guard let input = String(data: data, encoding: .utf8), !input.isEmpty else {
            print("")
            exit(0)
        }

        print(extractEntities(input).joined(separator: "\n"))
    }
}
