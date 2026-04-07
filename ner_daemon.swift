import Foundation

// [P3-1] Long-running NER daemon — reads lines from stdin, writes results to stdout
// Protocol: one JSON line per request, one JSON line per response
// Eliminates ~50-80ms subprocess spawn overhead per transcription
//
// Shared functions (isCleanTerm, extractEntities) live in ner_common.swift

@main
struct NERDaemon {
    static func main() {
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
    }
}
