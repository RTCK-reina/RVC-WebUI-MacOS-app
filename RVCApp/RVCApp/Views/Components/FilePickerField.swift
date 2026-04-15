import SwiftUI
import UniformTypeIdentifiers

/// Text field with a "Browse..." button for selecting a file or directory.
/// Wraps NSOpenPanel for native macOS semantics.
struct FilePickerField: View {
    let label: String
    @Binding var path: String
    var allowedContentTypes: [UTType] = [.audio]
    var chooseDirectory: Bool = false
    var initialDir: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            HStack {
                TextField("パス…", text: $path)
                    .textFieldStyle(.roundedBorder)
                Button("選択…") { openPanel() }
            }
        }
    }

    private func openPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = !chooseDirectory
        panel.canChooseDirectories = chooseDirectory
        panel.allowsMultipleSelection = false
        if !chooseDirectory {
            panel.allowedContentTypes = allowedContentTypes
        }
        if let initialDir, FileManager.default.fileExists(atPath: initialDir) {
            panel.directoryURL = URL(fileURLWithPath: initialDir)
        }
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}

/// Field supporting multiple selection (used for batch inputs).
struct MultiFilePickerField: View {
    let label: String
    @Binding var paths: [String]
    var allowedContentTypes: [UTType] = [.audio]
    var initialDir: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("追加…") { addFiles() }
                if !paths.isEmpty {
                    Button("クリア") { paths.removeAll() }
                        .foregroundStyle(.red)
                }
            }
            if paths.isEmpty {
                Text("ファイル未選択")
                    .font(.caption).foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, minHeight: 56)
                    .background(.quaternary, in: .rect(cornerRadius: 6))
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(paths, id: \.self) { p in
                            HStack {
                                Image(systemName: "waveform")
                                    .foregroundStyle(.secondary)
                                Text((p as NSString).lastPathComponent)
                                    .lineLimit(1).truncationMode(.middle)
                                Spacer()
                            }
                            .padding(.horizontal, 6).padding(.vertical, 2)
                        }
                    }
                }
                .frame(maxHeight: 140)
                .background(.quaternary, in: .rect(cornerRadius: 6))
            }
        }
    }

    private func addFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = allowedContentTypes
        if let initialDir, FileManager.default.fileExists(atPath: initialDir) {
            panel.directoryURL = URL(fileURLWithPath: initialDir)
        }
        if panel.runModal() == .OK {
            for url in panel.urls where !paths.contains(url.path) {
                paths.append(url.path)
            }
        }
    }
}
