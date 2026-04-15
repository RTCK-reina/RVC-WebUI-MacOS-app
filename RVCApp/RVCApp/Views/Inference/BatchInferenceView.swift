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

    @State private var selectedModel: String = ""
    @State private var dirPath: String = ""
    @State private var paths: [String] = []
    @State private var outputDir: String = ""
    @State private var f0UpKey: Double = 0
    @State private var f0Method: String = "rmvpe"
    @State private var indexRate: Double = 0.75
    @State private var filterRadius: Double = 3
    @State private var rmsMixRate: Double = 0.25
    @State private var protect: Double = 0.33
    @State private var format: String = "flac"

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
                .disabled(isRunning || selectedModel.isEmpty || (dirPath.isEmpty && paths.isEmpty))

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
        defer { isRunning = false }

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
            if r.status != "success" {
                errorMessage = r.info ?? "バッチ推論に失敗しました"
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func cancel() {
        let id = taskID
        Task {
            _ = try? await bridge.callRaw("cancel",
                params: .object(["task_id": .string(id)]))
        }
    }
}
