import SwiftUI

/// Response decoded from the `vc_single` RPC call.
struct VCSingleResult: Decodable {
    let status: String
    let info: String?
    let output_path: String?
    let sample_rate: Int?
}

struct SingleInferenceView: View {
    @EnvironmentObject var bridge: PythonBridge

    @State private var selectedModel: String = ""
    @State private var inputPath: String = ""
    @State private var f0UpKey: Double = 0
    @State private var f0Method: String = "rmvpe"
    @State private var indexRate: Double = 0.75
    @State private var filterRadius: Double = 3
    @State private var rmsMixRate: Double = 0.25
    @State private var protect: Double = 0.33
    @State private var resampleSR: Double = 0
    @State private var format: String = "flac"

    @State private var taskID: String = ""
    @State private var lastResult: VCSingleResult?
    @State private var errorMessage: String?
    @State private var isRunning = false

    private let f0Methods = ["pm", "harvest", "dio", "crepe", "rmvpe", "fcpe"]
    private let formats = ["flac", "wav", "mp3", "m4a"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                modelCard
                parametersCard
                actionCard
                if !taskID.isEmpty {
                    // No cancel button: single-file inference is a single
                    // PyTorch call and cannot be interrupted mid-flight.
                    ProgressBarView(taskID: taskID, title: "推論中")
                }
                if let err = errorMessage {
                    errorCard(err)
                }
                if let result = lastResult, let out = result.output_path {
                    resultCard(outputPath: out, info: result.info ?? "")
                }
            }
            .padding(16)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("単一推論")
                .font(.title2).bold()
            Text("音声ファイルを選んで、指定のモデルで変換します。結果は ~/Documents/RVC-WebUI/output/inference/ に保存されます。")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var modelCard: some View {
        GroupBox("モデル・入出力") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("モデル").font(.caption).foregroundStyle(.secondary)
                        .frame(width: 100, alignment: .leading)
                    Picker("", selection: $selectedModel) {
                        Text("（未選択）").tag("")
                        ForEach(bridge.models, id: \.self) { m in
                            Text(m).tag(m)
                        }
                    }
                    .labelsHidden()
                    Button("更新") {
                        Task { await refreshModels() }
                    }
                }
                FilePickerField(
                    label: "入力音声",
                    path: $inputPath,
                    initialDir: (bridge.initialInfo?["paths"]?["input_root"]?.stringValue).flatMap {
                        $0 + "/audio"
                    })
                HStack {
                    Picker("出力形式", selection: $format) {
                        ForEach(formats, id: \.self) { f in
                            Text(f.uppercased()).tag(f)
                        }
                    }
                    .frame(width: 180)
                    Spacer()
                }
            }
            .padding(8)
        }
    }

    private var parametersCard: some View {
        GroupBox("パラメータ") {
            VStack(alignment: .leading, spacing: 8) {
                sliderRow(label: "Pitch (半音)", value: $f0UpKey, range: -24...24, step: 1, format: "%.0f")
                Picker("F0 抽出", selection: $f0Method) {
                    ForEach(f0Methods, id: \.self) { m in Text(m).tag(m) }
                }
                .pickerStyle(.segmented)
                sliderRow(label: "Index Rate", value: $indexRate, range: 0...1, step: 0.01)
                sliderRow(label: "Filter Radius", value: $filterRadius, range: 0...7, step: 1, format: "%.0f")
                sliderRow(label: "RMS Mix Rate", value: $rmsMixRate, range: 0...1, step: 0.01)
                sliderRow(label: "Protect", value: $protect, range: 0...0.5, step: 0.01)
                sliderRow(label: "Resample (Hz)", value: $resampleSR, range: 0...48000, step: 1000, format: "%.0f")
            }
            .padding(8)
        }
    }

    private var actionCard: some View {
        HStack {
            Button(action: { Task { await run() } }) {
                Label("変換を実行", systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .keyboardShortcut(.return, modifiers: [.command])
            .disabled(isRunning || selectedModel.isEmpty || inputPath.isEmpty)
            .buttonStyle(.borderedProminent)
        }
    }

    private func sliderRow(
        label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        format: String = "%.2f"
    ) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
                .frame(width: 140, alignment: .leading)
            Slider(value: value, in: range, step: step)
            Text(String(format: format, value.wrappedValue))
                .font(.caption.monospacedDigit())
                .frame(width: 60, alignment: .trailing)
        }
    }

    private func errorCard(_ err: String) -> some View {
        Label(err, systemImage: "exclamationmark.triangle.fill")
            .padding(10)
            .foregroundStyle(.red)
            .background(.red.opacity(0.1), in: .rect(cornerRadius: 8))
    }

    private func resultCard(outputPath: String, info: String) -> some View {
        GroupBox("結果") {
            VStack(alignment: .leading, spacing: 8) {
                AudioPlayerView(url: URL(fileURLWithPath: outputPath))
                HStack {
                    Button("Finder で開く") {
                        NSWorkspace.shared.activateFileViewerSelecting(
                            [URL(fileURLWithPath: outputPath)])
                    }
                    Spacer()
                }
                if !info.isEmpty {
                    Text(info)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            .padding(8)
        }
    }

    // MARK: - RPC actions

    private func refreshModels() async {
        struct R: Decodable { let models: [String] }
        if let r: R = try? await bridge.call("list_models", params: JSONValue.object([:])) {
            bridge.models = r.models
        }
    }

    private func run() async {
        errorMessage = nil
        lastResult = nil
        let id = "vc_single_\(Int(Date().timeIntervalSince1970 * 1000))"
        taskID = id
        isRunning = true
        defer { isRunning = false }

        // Always call load_model; the backend is idempotent for the same
        // sid thanks to the load_vc cache (A-1 in PR #4), so a repeat call
        // costs ~50 ms with nothing to do. We keep the full response so we
        // can forward the resolved index_path to the vc_single call below
        // — previously both `file_index` and `file_index2` were hard-coded
        // to "" and the FAISS index was never actually consulted on the
        // Swift path, silently wasting the "Index Rate" slider the UI
        // exposes (A-5).
        let indexPath: String
        do {
            let loadResult = try await bridge.callRaw(
                "load_model",
                params: .object(["sid": .string(selectedModel)]))
            indexPath = loadResult["index_path"]?.stringValue ?? ""
        } catch {
            errorMessage = "モデルロード失敗: \(error.localizedDescription)"
            return
        }

        let params: JSONValue = .object([
            "task_id": .string(id),
            "sid": .string(selectedModel),
            "sid_index": .number(0),
            "input_audio_path": .string(inputPath),
            "f0_up_key": .number(f0UpKey),
            "f0_method": .string(f0Method),
            "file_index": .string(indexPath),
            "file_index2": .string(""),
            "index_rate": .number(indexRate),
            "filter_radius": .number(filterRadius),
            "resample_sr": .number(resampleSR),
            "rms_mix_rate": .number(rmsMixRate),
            "protect": .number(protect),
            "format": .string(format),
        ])

        do {
            let result: VCSingleResult = try await bridge.call(
                "vc_single", params: params, timeout: 1800)
            lastResult = result
            if result.status != "success" {
                errorMessage = result.info ?? "変換に失敗しました"
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

}
