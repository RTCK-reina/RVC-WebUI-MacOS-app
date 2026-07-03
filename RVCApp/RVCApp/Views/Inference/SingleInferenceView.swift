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
                return "rmvpe と高めの index_rate で音質を優先します。"
            case .balanced:
                return "既定値ベース。品質と速度の両立向けです。"
            case .speed:
                return "軽い F0 推定と低めの補正量で処理速度を優先します。"
            }
        }
    }

    @AppStorage("single.selectedModel") private var selectedModel: String = ""
    @AppStorage("single.inputPath") private var inputPath: String = ""
    @AppStorage("single.f0UpKey") private var f0UpKey: Double = 0
    @AppStorage("single.f0Method") private var f0Method: String = "rmvpe"
    @AppStorage("single.indexRate") private var indexRate: Double = 0.75
    @AppStorage("single.filterRadius") private var filterRadius: Double = 3
    @AppStorage("single.rmsMixRate") private var rmsMixRate: Double = 0.25
    @AppStorage("single.protect") private var protect: Double = 0.33
    @AppStorage("single.resampleSR") private var resampleSR: Double = 0
    @AppStorage("single.format") private var format: String = "flac"
    @AppStorage("single.preset") private var preset: InferencePreset = .balanced

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
                presetCard
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
        defer {
            isRunning = false
            taskID = ""
        }

        // Load model if different from previously loaded one.
        do {
            _ = try await bridge.callRaw(
                "load_model",
                params: .object(["sid": .string(selectedModel)]))
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
            "file_index": .string(""),
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

    private func applyPreset() {
        switch preset {
        case .quality:
            f0Method = "rmvpe"
            indexRate = 0.85
            filterRadius = 5
            rmsMixRate = 0.2
            protect = 0.35
            resampleSR = 0
            format = "flac"
        case .balanced:
            f0Method = "rmvpe"
            indexRate = 0.75
            filterRadius = 3
            rmsMixRate = 0.25
            protect = 0.33
            resampleSR = 0
            format = "flac"
        case .speed:
            f0Method = "dio"
            indexRate = 0.55
            filterRadius = 1
            rmsMixRate = 0.35
            protect = 0.2
            resampleSR = 32000
            format = "wav"
        }
    }

}
