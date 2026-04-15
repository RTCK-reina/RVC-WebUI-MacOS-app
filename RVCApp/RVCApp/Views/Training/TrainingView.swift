import SwiftUI

/// Full training pipeline view — preprocess → F0 extraction → train → index.
/// Provides "One-click" for running the entire pipeline in one RPC call.
struct TrainingView: View {
    @EnvironmentObject var bridge: PythonBridge

    // Experiment config
    @State private var expName: String = ""
    @State private var author: String = ""
    @State private var trainsetDir: String = ""
    @State private var srName: String = "40k"
    @State private var ifF0: Bool = true
    @State private var version: String = "v2"

    // Data processing
    @State private var f0Method: String = "rmvpe"
    @State private var nProcess: Double = 4
    @State private var gpus: String = "0"

    // Training params
    @State private var saveEpoch: Double = 5
    @State private var totalEpoch: Double = 200
    @State private var batchSize: Double = 4
    @State private var ifSaveLatest: Bool = true
    @State private var ifCacheGpu: Bool = false
    @State private var ifSaveEveryWeights: Bool = true
    @State private var spkId: Double = 0
    @State private var pretrainedG: String = ""
    @State private var pretrainedD: String = ""

    // Runtime state
    @State private var currentTaskID: String = ""
    @State private var logText: String = ""
    @State private var errorMsg: String?
    @State private var isRunning = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                experimentCard
                dataProcessingCard
                trainingParamsCard
                actionsCard
                if !currentTaskID.isEmpty {
                    ProgressBarView(
                        taskID: currentTaskID,
                        title: "処理中",
                        onCancel: cancel)
                }
                if let err = errorMsg {
                    Label(err, systemImage: "exclamationmark.triangle.fill")
                        .padding(10)
                        .foregroundStyle(.red)
                        .background(.red.opacity(0.1), in: .rect(cornerRadius: 8))
                }
                if !logText.isEmpty {
                    logView
                }
            }
            .padding(16)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("トレーニング").font(.title2).bold()
            Text("実験名を決めて、データ前処理 → F0抽出 → 学習 → インデックス構築 の順に進めます。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var experimentCard: some View {
        GroupBox("実験設定") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("実験名")
                        .frame(width: 100, alignment: .leading)
                    TextField("例: my_voice", text: $expName)
                        .textFieldStyle(.roundedBorder)
                }
                HStack {
                    Text("作者")
                        .frame(width: 100, alignment: .leading)
                    TextField("（任意）", text: $author)
                        .textFieldStyle(.roundedBorder)
                }
                FilePickerField(
                    label: "学習データディレクトリ（wav ファイル群）",
                    path: $trainsetDir,
                    chooseDirectory: true,
                    initialDir: defaultTrainingDir)
                HStack {
                    Picker("サンプリングレート", selection: $srName) {
                        Text("32k").tag("32k")
                        Text("40k").tag("40k")
                        Text("48k").tag("48k")
                    }
                    .frame(width: 220)
                    Picker("Version", selection: $version) {
                        Text("v1").tag("v1")
                        Text("v2").tag("v2")
                    }
                    .frame(width: 180)
                    Toggle("F0 (ピッチ学習) あり", isOn: $ifF0)
                }
            }
            .padding(8)
        }
    }

    private var dataProcessingCard: some View {
        GroupBox("データ処理") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Picker("F0 抽出手法", selection: $f0Method) {
                        ForEach(["pm", "harvest", "dio", "rmvpe"], id: \.self) {
                            Text($0).tag($0)
                        }
                    }
                    .frame(width: 260)
                    HStack {
                        Text("並列プロセス数").foregroundStyle(.secondary)
                        Stepper(value: $nProcess, in: 1...16, step: 1) {
                            Text("\(Int(nProcess))")
                                .frame(width: 40)
                        }
                    }
                }
                HStack {
                    Text("GPUs (CUDA: \"0-1\" 等、MPS/CPU は \"0\")")
                        .foregroundStyle(.secondary)
                    TextField("0", text: $gpus)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 100)
                }
                HStack {
                    Button("1. データ前処理") { Task { await runPreprocess() } }
                        .disabled(!canRunPhase)
                    Button("2. F0/特徴抽出") { Task { await runExtractF0() } }
                        .disabled(!canRunPhase)
                }
            }
            .padding(8)
        }
    }

    private var trainingParamsCard: some View {
        GroupBox("学習パラメータ") {
            VStack(alignment: .leading, spacing: 10) {
                sliderRow("保存エポック間隔", $saveEpoch, 1...50, 1, "%.0f")
                sliderRow("総エポック数", $totalEpoch, 10...5000, 10, "%.0f")
                sliderRow("バッチサイズ", $batchSize, 1...64, 1, "%.0f")
                sliderRow("スピーカー ID", $spkId, 0...4, 1, "%.0f")
                HStack {
                    Toggle("最新のみ保存", isOn: $ifSaveLatest)
                    Toggle("GPU にキャッシュ", isOn: $ifCacheGpu)
                    Toggle("毎回重みを保存", isOn: $ifSaveEveryWeights)
                }
                FilePickerField(
                    label: "Pretrained G (任意・空で自動解決)",
                    path: $pretrainedG,
                    allowedContentTypes: [],
                    initialDir: pretrainedDir)
                FilePickerField(
                    label: "Pretrained D (任意・空で自動解決)",
                    path: $pretrainedD,
                    allowedContentTypes: [],
                    initialDir: pretrainedDir)
                HStack {
                    Button("3. 学習開始") { Task { await runTrain() } }
                        .disabled(!canRunPhase)
                        .buttonStyle(.borderedProminent)
                    Button("4. インデックス学習") { Task { await runTrainIndex() } }
                        .disabled(!canRunPhase)
                }
            }
            .padding(8)
        }
    }

    private var actionsCard: some View {
        GroupBox("ワンクリック") {
            HStack {
                Button(action: { Task { await runAll() } }) {
                    Label("全パイプライン実行", systemImage: "bolt.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canRunPhase)
            }
            .padding(8)
        }
    }

    private var logView: some View {
        GroupBox("ログ") {
            ScrollView {
                Text(logText)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 240)
        }
    }

    // MARK: - Helpers

    private var canRunPhase: Bool {
        !isRunning && !expName.isEmpty && !trainsetDir.isEmpty
    }

    private var defaultTrainingDir: String? {
        (bridge.initialInfo?["paths"]?["input_root"]?.stringValue).flatMap {
            $0 + "/training"
        }
    }

    private var pretrainedDir: String? {
        bridge.initialInfo?["paths"]?["rmvpe_root"]?.stringValue.flatMap { root in
            // rmvpe sits next to pretrained_v2 in the bundle.
            String(root.split(separator: "/").dropLast().joined(separator: "/"))
        }
    }

    private func sliderRow(_ label: String, _ value: Binding<Double>, _ range: ClosedRange<Double>, _ step: Double, _ fmt: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
                .frame(width: 160, alignment: .leading)
            Slider(value: value, in: range, step: step)
            Text(String(format: fmt, value.wrappedValue))
                .font(.caption.monospacedDigit()).frame(width: 60, alignment: .trailing)
        }
    }

    // MARK: - Actions

    private func runPreprocess() async {
        await runStage(method: "preprocess", params: preprocessParams())
    }

    private func runExtractF0() async {
        await runStage(method: "extract_f0", params: extractF0Params())
    }

    private func runTrain() async {
        await runStage(method: "train", params: trainParams())
    }

    private func runTrainIndex() async {
        await runStage(method: "train_index", params: .object([
            "task_id": .string(makeTaskID("train_index")),
            "exp_name": .string(expName),
            "version": .string(version),
        ]))
    }

    private func runAll() async {
        // Merge all parameter objects since train_all reuses one params dict.
        var merged: [String: JSONValue] = [:]
        if case let .object(p1) = preprocessParams() { merged.merge(p1) { a, _ in a } }
        if case let .object(p2) = extractF0Params() { merged.merge(p2) { a, _ in a } }
        if case let .object(p3) = trainParams() { merged.merge(p3) { _, b in b } }
        merged["task_id"] = .string(makeTaskID("train_all"))
        merged["exp_name"] = .string(expName)
        merged["version"] = .string(version)
        await runStage(method: "train_all", params: .object(merged))
    }

    private func runStage(method: String, params: JSONValue) async {
        errorMsg = nil
        isRunning = true
        defer { isRunning = false }
        let id = params["task_id"]?.stringValue ?? makeTaskID(method)
        currentTaskID = id
        do {
            struct R: Decodable {
                let status: String
                let log: String?
                let error: String?
                let messages: [String]?
            }
            let r: R = try await bridge.call(method, params: params, timeout: 86400)
            if let l = r.log { logText = l }
            if let m = r.messages, !m.isEmpty {
                logText = (logText + "\n" + m.joined(separator: "\n")).trimmingCharacters(in: .whitespaces)
            }
            if r.status != "success" {
                errorMsg = r.error ?? "学習ステージ失敗: \(method)"
            }
        } catch {
            errorMsg = error.localizedDescription
        }
    }

    private func cancel() {
        let id = currentTaskID
        Task {
            _ = try? await bridge.callRaw("cancel",
                params: .object(["task_id": .string(id)]))
        }
    }

    private func makeTaskID(_ prefix: String) -> String {
        "\(prefix)_\(Int(Date().timeIntervalSince1970 * 1000))"
    }

    // MARK: - Parameter builders

    private func preprocessParams() -> JSONValue {
        .object([
            "task_id": .string(makeTaskID("preprocess")),
            "exp_name": .string(expName),
            "trainset_dir": .string(trainsetDir),
            "sr": .string(srName),
            "n_p": .number(nProcess),
        ])
    }

    private func extractF0Params() -> JSONValue {
        .object([
            "task_id": .string(makeTaskID("extract_f0")),
            "exp_name": .string(expName),
            "f0_method": .string(f0Method),
            "if_f0": .bool(ifF0),
            "version": .string(version),
            "gpus": .string(gpus),
            "n_p": .number(nProcess),
        ])
    }

    private func trainParams() -> JSONValue {
        .object([
            "task_id": .string(makeTaskID("train")),
            "exp_name": .string(expName),
            "sr": .string(srName),
            "if_f0": .bool(ifF0),
            "spk_id": .number(spkId),
            "save_epoch": .number(saveEpoch),
            "total_epoch": .number(totalEpoch),
            "batch_size": .number(batchSize),
            "if_save_latest": .bool(ifSaveLatest),
            "pretrained_G": .string(pretrainedG),
            "pretrained_D": .string(pretrainedD),
            "gpus": .string(gpus),
            "if_cache_gpu": .bool(ifCacheGpu),
            "if_save_every_weights": .bool(ifSaveEveryWeights),
            "version": .string(version),
            "author": .string(author),
        ])
    }
}
