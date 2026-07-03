import SwiftUI

struct VCMultiResult: Decodable {
    let status: String
    let output_dir: String?
    struct Item: Decodable {
        let input: String
        let output: String?
        let info: String
    }
    let results: [Item]?
    let info: String?
}

struct BatchInferenceView: View {
    @EnvironmentObject var bridge: PythonBridge

    private enum InferencePreset: String, CaseIterable, Identifiable {
        case quality
        case balanced
        case speed

        var id: String { rawValue }

        var label: String {
            switch self {
            case .quality: return "高品質"
            case .balanced: return "バランス"
            case .speed: return "高速"
            }
        }

        var description: String {
            switch self {
            case .quality:
                return "品質優先。補正量を高めて安定した声色再現を狙います。"
            case .balanced:
                return "標準運用向け。品質と処理時間のバランスを取ります。"
            case .speed:
                return "大量処理向け。軽量設定でスループットを優先します。"
            }
        }
    }

    @AppStorage("batch.selectedModel") private var selectedModel: String = ""
    @AppStorage("batch.dirPath") private var dirPath: String = ""
    @State private var paths: [String] = []  // session-specific file list
    @AppStorage("batch.outputDir") private var outputDir: String = ""
    @AppStorage("batch.f0UpKey") private var f0UpKey: Double = 0
    @AppStorage("batch.f0Method") private var f0Method: String = "rmvpe"
    @AppStorage("batch.indexRate") private var indexRate: Double = 0.75
    @AppStorage("batch.filterRadius") private var filterRadius: Double = 3
    @AppStorage("batch.rmsMixRate") private var rmsMixRate: Double = 0.25
    @AppStorage("batch.protect") private var protect: Double = 0.33
    @AppStorage("batch.format") private var format: String = "flac"
    @AppStorage("batch.preset") private var preset: InferencePreset = .balanced

    @State private var taskID = ""
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var lastResult: VCMultiResult?

    private let f0Methods = ["pm", "harvest", "dio", "crepe", "rmvpe", "fcpe"]
    private let formats = ["flac", "wav", "mp3", "m4a"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                presetCard
                GroupBox("モデル・入出力") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("モデル").font(.caption).foregroundStyle(.secondary)
                                .frame(width: 100, alignment: .leading)
                            Picker("", selection: $selectedModel) {
                                Text("（未選択）").tag("")
                                ForEach(bridge.models, id: \.self) { Text($0).tag($0) }
                            }
                            .labelsHidden()
                        }
                        FilePickerField(
                            label: "入力ディレクトリ（個別選択より優先）",
                            path: $dirPath,
                            chooseDirectory: true,
                            initialDir: defaultInputDir)
                        MultiFilePickerField(
                            label: "または個別ファイル",
                            paths: $paths,
                            initialDir: defaultInputDir)
                        FilePickerField(
                            label: "出力ディレクトリ（空欄で自動生成）",
                            path: $outputDir,
                            chooseDirectory: true,
                            initialDir: defaultOutputDir)
                        if let outputDirError {
                            Label(outputDirError, systemImage: "exclamationmark.triangle.fill")
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                        Picker("出力形式", selection: $format) {
                            ForEach(formats, id: \.self) { Text($0.uppercased()).tag($0) }
                        }
                        .frame(width: 220)
                    }
                    .padding(8)
                }

                GroupBox("パラメータ") {
                    VStack(alignment: .leading, spacing: 8) {
                        sliderRow("Pitch", $f0UpKey, -24...24, 1, "%.0f")
                        Picker("F0", selection: $f0Method) {
                            ForEach(f0Methods, id: \.self) { Text($0).tag($0) }
                        }
                        .pickerStyle(.segmented)
                        sliderRow("Index Rate", $indexRate, 0...1, 0.01)
                        sliderRow("Filter Radius", $filterRadius, 0...7, 1, "%.0f")
                        sliderRow("RMS Mix Rate", $rmsMixRate, 0...1, 0.01)
                        sliderRow("Protect", $protect, 0...0.5, 0.01)
                    }
                    .padding(8)
                }

                Button(action: { Task { await run() } }) {
                    Label("バッチ変換を実行", systemImage: "square.stack.3d.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning
                    || selectedModel.isEmpty
                    || (dirPath.isEmpty && paths.isEmpty)
                    || outputDirError != nil)

                if !taskID.isEmpty {
                    ProgressBarView(taskID: taskID, title: "バッチ推論中", onCancel: cancel)
                }
                if let err = errorMessage {
                    Label(err, systemImage: "exclamationmark.triangle.fill")
                        .padding(10)
                        .foregroundStyle(.red)
                        .background(.red.opacity(0.1), in: .rect(cornerRadius: 8))
                }
                if let result = lastResult, let items = result.results {
                    GroupBox("結果 (\(items.count) 件)") {
                        VStack(alignment: .leading, spacing: 4) {
                            if let dir = result.output_dir {
                                HStack {
                                    Text("出力先: \(dir)")
                                        .font(.caption.monospacedDigit())
                                    Spacer()
                                    Button("Finder で開く") {
                                        NSWorkspace.shared.activateFileViewerSelecting(
                                            [URL(fileURLWithPath: dir)])
                                    }
                                }
                            }
                            ScrollView {
                                LazyVStack(alignment: .leading, spacing: 2) {
                                    ForEach(items.indices, id: \.self) { i in
                                        let it = items[i]
                                        HStack {
                                            Image(systemName: it.output == nil ? "xmark.circle" : "checkmark.circle")
                                                .foregroundStyle(it.output == nil ? .red : .green)
                                            Text((it.input as NSString).lastPathComponent)
                                                .lineLimit(1)
                                        }
                                        .font(.caption)
                                    }
                                }
                            }
                            .frame(maxHeight: 200)
                        }
                        .padding(8)
                    }
                }
            }
            .padding(16)
        }
    }

    private var presetCard: some View {
        GroupBox("品質 / 速度プリセット") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("目的", selection: $preset) {
                    ForEach(InferencePreset.allCases) { p in
                        Text(p.label).tag(p)
                    }
                }
                .pickerStyle(.segmented)

                Text(preset.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Button("推奨値を適用") {
                    applyPreset()
                }
                .buttonStyle(.bordered)
            }
            .padding(8)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("バッチ推論").font(.title2).bold()
            Text("ディレクトリまたは複数ファイルをまとめて変換します。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var defaultInputDir: String? {
        (bridge.initialInfo?["paths"]?["input_root"]?.stringValue).flatMap { $0 + "/audio" }
    }

    private var defaultOutputDir: String? {
        bridge.initialInfo?["paths"]?["output_root"]?.stringValue
    }

    private var outputDirError: String? {
        PathValidation.outputRootError(
            outputDir,
            root: defaultOutputDir,
            label: "出力ディレクトリ")
    }

    private func sliderRow(_ label: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, _ step: Double, _ fmt: String = "%.2f") -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
                .frame(width: 130, alignment: .leading)
            Slider(value: value, in: range, step: step)
            Text(String(format: fmt, value.wrappedValue))
                .font(.caption.monospacedDigit()).frame(width: 60, alignment: .trailing)
        }
    }

    private func run() async {
        errorMessage = nil
        lastResult = nil
        let id = "vc_multi_\(Int(Date().timeIntervalSince1970 * 1000))"
        taskID = id
        isRunning = true
        defer {
            isRunning = false
            taskID = ""
        }

        do {
            _ = try await bridge.callRaw("load_model",
                params: .object(["sid": .string(selectedModel)]))
        } catch {
            errorMessage = "モデルロード失敗: \(error.localizedDescription)"
            return
        }

        let params: JSONValue = .object([
            "task_id": .string(id),
            "sid": .string(selectedModel),
            "sid_index": .number(0),
            "dir_path": .string(dirPath),
            "paths": .array(paths.map { .string($0) }),
            "output_dir": .string(outputDir),
            "f0_up_key": .number(f0UpKey),
            "f0_method": .string(f0Method),
            "file_index": .string(""),
            "file_index2": .string(""),
            "index_rate": .number(indexRate),
            "filter_radius": .number(filterRadius),
            "resample_sr": .number(0),
            "rms_mix_rate": .number(rmsMixRate),
            "protect": .number(protect),
            "format": .string(format),
        ])

        do {
            let r: VCMultiResult = try await bridge.call("vc_multi", params: params, timeout: 7200)
            lastResult = r
            if r.status == "success" {
                AppNotification.send(
                    title: "バッチ推論が完了しました",
                    body: "全ファイルの推論が正常に終了しました。")
            } else if r.status == "cancelled" {
                // User-initiated stop — not a failure. No red banner.
                errorMessage = nil
                AppNotification.send(
                    title: "バッチ推論を停止しました",
                    body: "ユーザー操作により中断されました。")
            } else {
                errorMessage = r.info ?? "バッチ推論に失敗しました"
                AppNotification.send(
                    title: "バッチ推論に失敗しました",
                    body: errorMessage ?? "")
            }
        } catch {
            errorMessage = error.localizedDescription
            AppNotification.send(
                title: "バッチ推論に失敗しました",
                body: error.localizedDescription)
        }
    }

    private func cancel() {
        let id = taskID
        Task {
            // Staged escalation (soft → force → hardRestart) so a wedged
            // backend is actually recovered and the user is told if even that
            // fails, instead of silently clearing the card.
            let outcome = await bridge.cancelTask(id)
            bridge.clearProgress(for: id)
            if case .restartFailed(let reason) = outcome {
                errorMessage = "停止に失敗しました（バックエンド再起動も失敗）: \(reason)"
            }
        }
    }

    private func applyPreset() {
        switch preset {
        case .quality:
            f0Method = "rmvpe"
            indexRate = 0.85
            filterRadius = 5
            rmsMixRate = 0.2
            protect = 0.35
            format = "flac"
        case .balanced:
            f0Method = "rmvpe"
            indexRate = 0.75
            filterRadius = 3
            rmsMixRate = 0.25
            protect = 0.33
            format = "flac"
        case .speed:
            f0Method = "dio"
            indexRate = 0.55
            filterRadius = 1
            rmsMixRate = 0.35
            protect = 0.2
            format = "wav"
        }
    }
}
