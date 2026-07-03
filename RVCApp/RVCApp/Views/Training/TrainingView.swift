import SwiftUI

/// Full training pipeline view — preprocess → F0 extraction → train → index.
/// Provides "One-click" for running the entire pipeline in one RPC call.
struct TrainingView: View {
    @EnvironmentObject var bridge: PythonBridge

    private enum TrainingPreset: String, CaseIterable, Identifiable {
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
                return "音質重視。F0 精度を優先し、保存頻度を上げて安定した学習経路を確保します。"
            case .balanced:
                return "日常運用向け。品質と速度のバランスが取れた無難な設定です。"
            case .speed:
                return "反復重視。高速に回して実験回数を増やすための設定です。"
            }
        }
    }

    // Experiment config
    @AppStorage("train.expName") private var expName: String = ""
    @AppStorage("train.author") private var author: String = ""
    @AppStorage("train.trainsetDir") private var trainsetDir: String = ""
    @AppStorage("train.srName") private var srName: String = "40k"
    @AppStorage("train.ifF0") private var ifF0: Bool = true
    @AppStorage("train.version") private var version: String = "v2"

    // Data processing
    @AppStorage("train.f0Method") private var f0Method: String = "rmvpe"
    @AppStorage("train.nProcess") private var nProcess: Double =
        Double(max(1, ProcessInfo.processInfo.activeProcessorCount - 1))
    private let gpus: String = "0"  // Always "0" on macOS (MPS/CPU)

    // Training params
    @AppStorage("train.saveEpoch") private var saveEpoch: Double = 5
    @AppStorage("train.totalEpoch") private var totalEpoch: Double = 200
    @AppStorage("train.batchSize") private var batchSize: Double = 4
    @AppStorage("train.ifSaveLatest") private var ifSaveLatest: Bool = true
    @AppStorage("train.ifCacheGpu") private var ifCacheGpu: Bool = false
    @AppStorage("train.ifSaveEveryWeights") private var ifSaveEveryWeights: Bool = true
    @AppStorage("train.requireGPU") private var requireGPU: Bool = false
    @AppStorage("train.spkId") private var spkId: Double = 0
    @AppStorage("train.pretrainedG") private var pretrainedG: String = ""
    @AppStorage("train.pretrainedD") private var pretrainedD: String = ""
    @AppStorage("train.preset") private var preset: TrainingPreset = .balanced

    // Runtime state
    @State private var currentTaskID: String = ""
    @State private var logText: String = ""
    @State private var errorMsg: String?
    @State private var isRunning = false
    @State private var cancelDisplay: CancelDisplay = .idle

    /// Visible state of the staged cancel flow. Reflects the outcome of
    /// `PythonBridge.cancelTask` so the user can distinguish "stopped
    /// cleanly" from "we had to SIGKILL the whole backend".
    private enum CancelDisplay: Equatable {
        case idle
        case cancelling       // 停止 RPC 送信中
        case forceCancelling  // force=true へエスカレーション
        case restarting       // hardRestart 実行中
        case cancelled        // 停止完了（直近メッセージ）
        case restarted        // バックエンド再起動で復帰
        case failed(String)   // 例外等で停止不能

        var bannerMessage: String? {
            switch self {
            case .idle: return nil
            case .cancelling: return "停止中..."
            case .forceCancelling: return "応答が遅いため強制停止を送信中..."
            case .restarting: return "バックエンドを再起動中..."
            case .cancelled: return "停止しました"
            case .restarted: return "バックエンドを再起動して復帰しました"
            case .failed(let s): return "停止失敗: \(s)"
            }
        }

        var bannerTint: Color {
            switch self {
            case .cancelled, .restarted: return .green
            case .failed: return .red
            case .idle: return .clear
            default: return .orange
            }
        }

        var bannerIcon: String {
            switch self {
            case .failed: return "exclamationmark.triangle.fill"
            case .cancelled, .restarted: return "checkmark.circle"
            default: return "stop.circle"
            }
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                presetCard
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
                if let msg = cancelDisplay.bannerMessage {
                    Label(msg, systemImage: cancelDisplay.bannerIcon)
                        .padding(10)
                        .foregroundStyle(cancelDisplay.bannerTint)
                        .background(cancelDisplay.bannerTint.opacity(0.1), in: .rect(cornerRadius: 8))
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

    private var presetCard: some View {
        GroupBox("品質 / 速度プリセット") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("目的", selection: $preset) {
                    ForEach(TrainingPreset.allCases) { p in
                        Text(p.label).tag(p)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: preset) { _ in applyTrainingPreset() }

                Text(preset.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(8)
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
                if let expNameError {
                    Label(expNameError, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
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
                        let maxProcs = ProcessInfo.processInfo.activeProcessorCount
                        Text("並列プロセス数 (最大 \(maxProcs))").foregroundStyle(.secondary)
                        Stepper(value: $nProcess, in: 1...Double(maxProcs), step: 1) {
                            Text("\(Int(nProcess))")
                                .frame(width: 40)
                        }
                    }
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
                    Toggle("GPU 学習を要求", isOn: $requireGPU)
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
        !isRunning && expNameError == nil && !trainsetDir.isEmpty
    }

    private var expNameError: String? {
        PathValidation.leafNameError(expName, label: "実験名")
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

    private func applyTrainingPreset() {
        let totalCPU = ProcessInfo.processInfo.activeProcessorCount
        let defaultProcs = max(1, totalCPU - 1)
        let gpuBackend = bridge.resourceStats.gpuBackend
        let gpuEnabled = (gpuBackend == "mps" || gpuBackend == "cuda")

        switch preset {
        case .quality:
            f0Method = "rmvpe"
            nProcess = Double(max(1, defaultProcs - 1))
            saveEpoch = 5
            totalEpoch = max(totalEpoch, 300)
            batchSize = gpuEnabled ? 4 : 2
            ifSaveLatest = false
            ifCacheGpu = gpuEnabled
            ifSaveEveryWeights = true
        case .balanced:
            f0Method = "rmvpe"
            nProcess = Double(defaultProcs)
            saveEpoch = 5
            totalEpoch = max(200, min(totalEpoch, 400))
            batchSize = gpuEnabled ? 4 : 2
            ifSaveLatest = true
            ifCacheGpu = false
            ifSaveEveryWeights = true
        case .speed:
            f0Method = ifF0 ? "fcpe" : "pm"
            nProcess = Double(totalCPU)
            saveEpoch = 10
            totalEpoch = min(totalEpoch, 180)
            batchSize = gpuEnabled ? 8 : 3
            ifSaveLatest = true
            ifCacheGpu = gpuEnabled
            ifSaveEveryWeights = false
        }
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
        defer {
            isRunning = false
            currentTaskID = ""
        }
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
            if r.status == "success" {
                AppNotification.send(
                    title: "トレーニング完了",
                    body: "\(method) が正常に終了しました。")
            } else if r.status == "cancelled" {
                // User-initiated stop. Cancellation is owned by the
                // cancelDisplay state machine (see cancel()); setting errorMsg
                // or firing a failure notification here would render a
                // contradictory green "停止しました" + red "失敗" at once.
            } else {
                errorMsg = r.error ?? "学習ステージ失敗: \(method)"
                AppNotification.send(
                    title: "トレーニング失敗",
                    body: errorMsg ?? method)
            }
        } catch {
            errorMsg = error.localizedDescription
            AppNotification.send(
                title: "トレーニング失敗",
                body: error.localizedDescription)
        }
    }

    private func cancel() {
        let id = currentTaskID
        guard !id.isEmpty else { return }
        cancelDisplay = .cancelling
        // Clear progress immediately so the UI stops re-rendering stale
        // percentages while the multi-stage cancel unfolds. cancelledTaskIDs
        // is used on the bridge side to ignore late progress notifications.
        bridge.clearProgress(for: id)
        Task {
            // Stage-1 timeout (2s) covers the common case where the
            // backend is blocked on the tail log but will respond once
            // SIGTERM lands. Stage 2 (force=true) is surfaced here so
            // the user sees it happen; the bridge escalates internally.
            let stageTwoDelay: UInt64 = 2_100_000_000  // 2.1s
            let flipToForce = Task { @MainActor in
                try? await Task.sleep(nanoseconds: stageTwoDelay)
                if cancelDisplay == .cancelling {
                    cancelDisplay = .forceCancelling
                }
            }
            let flipToRestart = Task { @MainActor in
                try? await Task.sleep(nanoseconds: stageTwoDelay + 1_000_000_000)
                if cancelDisplay == .forceCancelling {
                    cancelDisplay = .restarting
                }
            }

            let outcome = await bridge.cancelTask(id)
            flipToForce.cancel()
            flipToRestart.cancel()

            switch outcome {
            case .graceful, .forced:
                cancelDisplay = .cancelled
            case .restarted:
                cancelDisplay = .restarted
            case .restartFailed(let reason):
                // Everything we tried failed — the Python backend is down.
                // Surface the real error instead of a success banner so
                // the user knows to check logs / restart manually.
                cancelDisplay = .failed(reason)
            }
            // Clear the banner after a brief visible interval, but
            // leave .failed up so the user can read the reason.
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            if cancelDisplay == .cancelled || cancelDisplay == .restarted {
                cancelDisplay = .idle
            }
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
            "require_gpu": .bool(requireGPU),
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
            "require_gpu": .bool(requireGPU),
            "version": .string(version),
            "author": .string(author),
        ])
    }
}
