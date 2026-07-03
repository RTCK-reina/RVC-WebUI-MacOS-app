import SwiftUI

struct ContentView: View {
    @EnvironmentObject var bridge: PythonBridge
    @EnvironmentObject var appState: AppState

    var body: some View {
        NavigationSplitView {
            Sidebar(selection: $appState.selection)
        } detail: {
            VStack(spacing: 0) {
                Group {
                    switch appState.selection {
                    case .inferenceSingle: SingleInferenceView()
                    case .inferenceBatch:  BatchInferenceView()
                    case .separation:      SeparationView()
                    case .training:        TrainingView()
                    case .modelsCompare:   ModelCompareView()
                    case .modelsMerge:     ModelMergeView()
                    case .modelsInfo:      ModelInfoView()
                    case .modelsExtract:   ModelExtractView()
                    case .onnxExport:      OnnxExportView()
                    case .realtimeVC:      RealtimeVCView()
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                if !bridge.activeProgress.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("進行中タスク (\(bridge.activeProgress.count))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    bridge.taskListMinimized.toggle()
                                }
                            } label: {
                                Image(systemName: bridge.taskListMinimized
                                      ? "chevron.up" : "chevron.down")
                                    .font(.caption)
                            }
                            .buttonStyle(.borderless)
                            .help(bridge.taskListMinimized ? "展開" : "最小化")
                        }
                        if !bridge.taskListMinimized {
                            ActiveTasksList()
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.top, 8)
                    .padding(.bottom, 6)
                    .background(.ultraThinMaterial)
                }

                Divider()
                StatusBarView()
            }
        }
        .overlay {
            if appState.isLaunching {
                LaunchingOverlay()
            } else if let err = appState.launchError {
                LaunchErrorOverlay(message: err)
            }
        }
    }
}

private struct Sidebar: View {
    @Binding var selection: NavigationItem

    private let items: [(String, [NavigationItem])] = [
        ("推論",     [.inferenceSingle, .inferenceBatch]),
        ("音声処理", [.separation, .realtimeVC]),
        ("学習",     [.training]),
        ("モデル",   [.modelsCompare, .modelsMerge, .modelsInfo, .modelsExtract, .onnxExport]),
    ]

    var body: some View {
        List(selection: $selection) {
            ForEach(items, id: \.0) { section in
                Section(section.0) {
                    ForEach(section.1) { item in
                        Label(item.title, systemImage: item.systemImage)
                            .tag(item)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .frame(minWidth: 200)
    }
}

private struct LaunchingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.35).ignoresSafeArea()
            VStack(spacing: 12) {
                ProgressView()
                Text("Python バックエンド起動中…")
                    .foregroundStyle(.white)
            }
            .padding(24)
            .background(.black.opacity(0.6), in: .rect(cornerRadius: 12))
        }
    }
}

private struct LaunchErrorOverlay: View {
    let message: String

    private var logFileURL: URL? {
        FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Logs/\(AppState.userDirectoryName)/bridge.log")
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(alignment: .leading, spacing: 12) {
                Label("起動に失敗しました", systemImage: "exclamationmark.triangle")
                    .font(.headline)
                ScrollView {
                    Text(message)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 280)
                HStack {
                    if let url = logFileURL {
                        Button("ログをFinderで開く") {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        }
                    }
                    Button("エラーをコピー") {
                        let pb = NSPasteboard.general
                        pb.clearContents()
                        pb.setString(message, forType: .string)
                    }
                    Spacer()
                    Button("終了") { NSApplication.shared.terminate(nil) }
                        .keyboardShortcut(.escape)
                }
            }
            .foregroundStyle(.white)
            .padding(20)
            .background(.red.opacity(0.9), in: .rect(cornerRadius: 12))
            .frame(maxWidth: 720)
        }
    }
}
