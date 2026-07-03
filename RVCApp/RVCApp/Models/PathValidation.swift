import Foundation

enum PathValidation {
    static func leafNameError(_ value: String, label: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "\(label)を入力してください" }
        if trimmed.count > 120 { return "\(label)は120文字以内にしてください" }
        if trimmed == "." || trimmed == ".." { return "\(label)に . や .. は使えません" }
        if trimmed.contains("/") || trimmed.contains("\\") {
            return "\(label)にはフォルダ区切りを含められません"
        }
        if trimmed.unicodeScalars.contains(where: {
            CharacterSet.controlCharacters.contains($0)
        }) {
            return "\(label)に制御文字は使えません"
        }
        return nil
    }

    static func outputRootError(_ value: String, root: String?, label: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return nil }
        guard let root, !root.isEmpty else { return nil }
        let expanded = (trimmed as NSString).expandingTildeInPath
        if !expanded.hasPrefix("/") { return nil }
        // Collapse "." / ".." components before the containment check so that
        // "/root/sub/../../etc" cannot slip past the prefix comparison.
        let standardized = URL(fileURLWithPath: expanded).standardizedFileURL.path
        let expandedRoot = URL(
            fileURLWithPath: (root as NSString).expandingTildeInPath
        ).standardizedFileURL.path
        if standardized == expandedRoot || standardized.hasPrefix(expandedRoot + "/") {
            return nil
        }
        return "\(label)は \(expandedRoot) 配下を指定してください"
    }
}
