import SwiftUI

/// Backend payload returned from list_audio_devices.
struct AudioDeviceListing: Decodable {
    struct Device: Decodable, Hashable {
        let index: Int
        let name: String
        let hostapi: String
        let max_channels: Int?
        let default_sr: Double?
    }
    let host_apis: [String]?
    let input: [Device]
    let output: [Device]
}

struct RealtimeVCView: View {
    @EnvironmentObject var bridge: PythonBridge

    // Device / model
    @State private var devices: AudioDeviceListing = .init(host_apis: [], input: [], output: [])
    @AppStorage("rt.selectedModel") private var selectedModel: String = ""
    @AppStorage("rt.indexPath") private var indexPath: String = ""
    @AppStorage("rt.inputDeviceIndex") private var inputDeviceIndex: Int = -1
    @AppStorage("rt.outputDeviceIndex") private var outputDeviceIndex: Int = -1
    @AppStorage("rt.sampleRate") private var sampleRate: Double = 48000

    // Parameters (hot-reloadable)
    @AppStorage("rt.pitch") private var pitch: Double = 0
    @AppStorage("rt.formant") private var formant: Double = 0
    @AppStorage("rt.indexRate") private var indexRate: Double = 0.0
    @AppStorage("rt.threshold") private var threshold: Double = -60
    @AppStorage("rt.blockTime") private var blockTime: Double = 0.25
    @AppStorage("rt.f0Method") private var f0Method: String = "fcpe"
    @AppStorage("rt.protect") private var protect: Double = 0.33

    // State
    @State private var isRunning = false
    @State private var errorMsg: String?
    @State private var lastInfo: String = ""
    @State private var eventLogExpanded: Bool = false

    private let f0Methods = ["pm", "harvest", "dio", "crepe", "rmvpe", "fcpe"]

    private static let logTimeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                modelCard
                deviceCard
                parameterCard
                controlsCard
                if let err = errorMsg {
                    Label(err, systemImage: "exclamationmark.triangle.fill")
                        .padding(10)
                        .foregroundStyle(.red)
                        .background(.red.opacity(0.1), in: .rect(cornerRadius: 8))
                }
                if isRunning {
                    diagnosticsStatusCard
                    meterCard
                    metricsCard
                }
                if isRunning || !bridge.realtimeEvents.isEmpty {
                    eventLogCard
                }
                if !lastInfo.isEmpty {
                    Text(lastInfo)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(16)
            .task(id: bridge.isReady) {
                if bridge.isReady { await refreshDevices() }
            }
            .onChange(of: bridge.realtimeRunning) { running in
                // The backend is authoritative: a fatal realtime_error (after
                // repeated callback failures) or an external stop flips this
                // false. Leave the running layout instead of getting stuck in
                // a dead-but-"running" UI with disabled controls.
                if !running && isRunning {
                    isRunning = false
                    if let le = bridge.lastError {
                        errorMsg = le
                    }
                }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("リアルタイム音声変換").font(.title2).bold()
            Text("マイク入力をリアルタイムで変換し、選択した出力デバイスに流します。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var modelCard: some View {
        GroupBox("モデル") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("モデル").frame(width: 100, alignment: .leading)
                    Picker("", selection: $selectedModel) {
                        Text("（未選択）").tag("")
                        ForEach(bridge.models, id: \.self) { Text($0).tag($0) }
                    }
                    .labelsHidden()
                }
                FilePickerField(
                    label: "インデックス (.index 任意)",
                    path: $indexPath,
                    allowedContentTypes: [])
            }
            .padding(8)
        }
        .disabled(isRunning)
    }

    private var deviceCard: some View {
        GroupBox("オーディオデバイス") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("入力").frame(width: 100, alignment: .leading)
                    Picker("", selection: $inputDeviceIndex) {
                        Text("自動").tag(-1)
                        ForEach(devices.input, id: \.index) { d in
                            Text("\(d.name) [\(d.hostapi)]")
                                .tag(d.index)
                        }
                    }
                    .labelsHidden()
                }
                HStack {
                    Text("出力").frame(width: 100, alignment: .leading)
                    Picker("", selection: $outputDeviceIndex) {
                        Text("自動").tag(-1)
                        ForEach(devices.output, id: \.index) { d in
                            Text("\(d.name) [\(d.hostapi)]")
                                .tag(d.index)
                        }
                    }
                    .labelsHidden()
                }
                HStack {
                    Text("サンプリングレート").frame(width: 150, alignment: .leading)
                    Picker("", selection: $sampleRate) {
                        Text("44100 Hz").tag(44100.0)
                        Text("48000 Hz").tag(48000.0)
                        Text("32000 Hz").tag(32000.0)
                    }
                    .labelsHidden()
                    .frame(width: 180)
                    Button("デバイス再取得") {
                        Task { await refreshDevices() }
                    }
                }
            }
            .padding(8)
        }
        .disabled(isRunning)
    }

    private var parameterCard: some View {
        GroupBox("リアルタイムパラメータ") {
            VStack(alignment: .leading, spacing: 8) {
                sliderRow("Pitch", $pitch, -24...24, 1, "%.0f", hotReload: true)
                sliderRow("Formant", $formant, -5...5, 0.01, hotReload: true)
                sliderRow("Index Rate", $indexRate, 0...1, 0.01, hotReload: true)
                sliderRow("Threshold (dB)", $threshold, -80...0, 1, "%.0f", hotReload: false)
                sliderRow("Block (秒)", $blockTime, 0.05...1.5, 0.01, hotReload: false)
                sliderRow("Protect", $protect, 0...0.5, 0.01, hotReload: false)
                HStack {
                    Text("F0 手法").foregroundStyle(.secondary)
                    Picker("", selection: $f0Method) {
                        ForEach(f0Methods, id: \.self) { Text($0).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(isRunning)
                }
            }
            .padding(8)
        }
    }

    private var controlsCard: some View {
        HStack {
            if isRunning {
                Button(action: { Task { await stop() } }) {
                    Label("停止", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            } else {
                Button(action: { Task { await start() } }) {
                    Label("リアルタイム開始", systemImage: "mic.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedModel.isEmpty)
            }
        }
    }

    private func sliderRow(
        _ label: String,
        _ value: Binding<Double>,
        _ range: ClosedRange<Double>,
        _ step: Double,
        _ fmt: String = "%.2f",
        hotReload: Bool = false
    ) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
                .frame(width: 140, alignment: .leading)
            Slider(value: value, in: range, step: step, onEditingChanged: { editing in
                if hotReload && !editing && isRunning {
                    Task { await pushParamUpdate() }
                }
            })
            Text(String(format: fmt, value.wrappedValue))
                .font(.caption.monospacedDigit()).frame(width: 60, alignment: .trailing)
        }
    }

    // MARK: - Diagnostics cards

    /// Top-line health summary — at a glance, is the session OK, gated, clipping, behind?
    private var diagnosticsStatusCard: some View {
        let m = bridge.realtimeMetrics
        return GroupBox("状態") {
            HStack(spacing: 8) {
                badge(
                    text: m.errorCount > 0 ? "推論エラー" : "推論OK",
                    color: m.errorCount > 0 ? .red : .green)
                badge(
                    text: m.isGated ? "無音ゲート中" : "音声あり",
                    color: m.isGated ? .orange : .blue)
                switch m.latencyHealth {
                case .ok:       badge(text: "レイテンシOK", color: .green)
                case .warn:     badge(text: "レイテンシ警告", color: .yellow)
                case .critical: badge(text: "レイテンシ不足", color: .red)
                case .unknown:  badge(text: "計測中", color: .gray)
                }
                if m.inputClipped { badge(text: "入力クリップ", color: .red) }
                if m.outputClipped { badge(text: "出力クリップ", color: .red) }
                Spacer()
            }
            .padding(8)
        }
    }

    private func badge(text: String, color: Color) -> some View {
        Text(text)
            .font(.caption).bold()
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(color.opacity(0.18), in: .capsule)
            .foregroundStyle(color)
            .overlay(Capsule().stroke(color.opacity(0.4), lineWidth: 0.5))
    }

    /// VU meters for input and output. The threshold marker on the input row
    /// shows where the noise gate kicks in, so the user can see whether their
    /// signal is above or below it.
    private var meterCard: some View {
        let m = bridge.realtimeMetrics
        return GroupBox("シグナル") {
            VStack(alignment: .leading, spacing: 10) {
                meterRow(
                    label: "入力",
                    rmsDB: m.inputRMSdB,
                    peakDB: m.inputPeakdB,
                    clipped: m.inputClipped,
                    gated: m.isGated,
                    thresholdDB: m.thresholdDB)
                meterRow(
                    label: "出力",
                    rmsDB: m.outputRMSdB,
                    peakDB: m.outputPeakdB,
                    clipped: m.outputClipped,
                    gated: false,
                    thresholdDB: nil)
                if m.inputRMSdB < -55 && !m.isGated {
                    Label(
                        "マイク信号がほぼ無音です。デバイス選択・接続・OSのマイク権限を確認してください。",
                        systemImage: "questionmark.circle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                if m.outputRMSdB < -55 && m.inputRMSdB > -40 && !m.isGated {
                    Label(
                        "入力は来ていますが出力が無音です。モデルロード・Protect値・推論エラーを確認してください。",
                        systemImage: "speaker.slash")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
            .padding(8)
        }
    }

    private func meterRow(
        label: String, rmsDB: Double, peakDB: Double,
        clipped: Bool, gated: Bool, thresholdDB: Double?
    ) -> some View {
        let minDB: Double = -60
        let maxDB: Double = 0
        let rmsNorm = max(0, min(1, (rmsDB - minDB) / (maxDB - minDB)))
        let peakNorm = max(0, min(1, (peakDB - minDB) / (maxDB - minDB)))
        return HStack(spacing: 8) {
            Text(label).font(.caption).bold()
                .frame(width: 36, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Rectangle().fill(Color.secondary.opacity(0.15))
                    LinearGradient(
                        colors: [.green, .green, .yellow, .red],
                        startPoint: .leading, endPoint: .trailing)
                        .frame(width: max(0, geo.size.width * rmsNorm))
                        .opacity(gated ? 0.25 : 1.0)
                    // Peak hold line — red on clip.
                    Rectangle()
                        .fill(clipped ? Color.red : Color.white.opacity(0.7))
                        .frame(width: 2)
                        .offset(x: max(0, geo.size.width * peakNorm - 1))
                    // Threshold marker (input row only).
                    if let t = thresholdDB {
                        let tNorm = max(0, min(1, (t - minDB) / (maxDB - minDB)))
                        Rectangle()
                            .fill(Color.orange)
                            .frame(width: 1)
                            .offset(x: geo.size.width * tNorm)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 3))
            }
            .frame(height: 14)
            Text(String(format: "%5.1f dB", rmsDB))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 70, alignment: .trailing)
        }
    }

    /// Numeric readouts for latency, over/underruns, and device info. Shows
    /// contextual hints underneath when a condition is bad enough to act on.
    private var metricsCard: some View {
        let m = bridge.realtimeMetrics
        return GroupBox("数値メトリクス") {
            VStack(alignment: .leading, spacing: 8) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    GridRow {
                        Text("推論時間").foregroundStyle(.secondary)
                        Text(String(format: "%.1f ms", m.inferenceMs))
                            .foregroundStyle(latencyColor(m))
                            .monospacedDigit()
                        Text("ブロック予算").foregroundStyle(.secondary)
                        Text(String(format: "%.1f ms", m.blockBudgetMs))
                            .monospacedDigit()
                    }
                    GridRow {
                        Text("コールバック時間").foregroundStyle(.secondary)
                        Text(String(format: "%.1f ms", m.blockMs))
                            .monospacedDigit()
                        Text("推論エラー").foregroundStyle(.secondary)
                        Text("\(m.errorCount)")
                            .monospacedDigit()
                            .foregroundStyle(m.errorCount > 0 ? .red : .primary)
                    }
                    GridRow {
                        Text("入力 under/overflow").foregroundStyle(.secondary)
                        Text("\(m.inputUnderflow) / \(m.inputOverflow)")
                            .monospacedDigit()
                            .foregroundStyle(
                                (m.inputUnderflow + m.inputOverflow) > 0 ? .orange : .primary)
                        Text("出力 under/overflow").foregroundStyle(.secondary)
                        Text("\(m.outputUnderflow) / \(m.outputOverflow)")
                            .monospacedDigit()
                            .foregroundStyle(
                                (m.outputUnderflow + m.outputOverflow) > 0 ? .orange : .primary)
                    }
                    GridRow {
                        Text("入力デバイス").foregroundStyle(.secondary)
                        Text(m.inputDeviceName.isEmpty ? "—" : m.inputDeviceName)
                            .lineLimit(1).truncationMode(.middle)
                        Text("出力デバイス").foregroundStyle(.secondary)
                        Text(m.outputDeviceName.isEmpty ? "—" : m.outputDeviceName)
                            .lineLimit(1).truncationMode(.middle)
                    }
                }
                .font(.caption)

                if m.latencyHealth == .critical {
                    Label(
                        "推論が間に合っていません。Block(秒) を大きく、軽い F0 手法 (fcpe/pm)、Protect を下げる、GPU 有効化などを検討してください。",
                        systemImage: "bolt.slash.fill")
                        .font(.caption).foregroundStyle(.red)
                } else if m.latencyHealth == .warn {
                    Label(
                        "推論時間がブロック予算に迫っています。負荷が上がると途切れる可能性があります。",
                        systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.yellow)
                }
                if m.totalUnderOverflow >= 5 {
                    Label(
                        "sounddevice が under/overrun を検出しています。サンプルレートを下げる、Block(秒) を大きく、他の重いアプリを閉じる等を試してください。",
                        systemImage: "exclamationmark.circle")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            .padding(8)
        }
    }

    private func latencyColor(_ m: RealtimeMetrics) -> Color {
        switch m.latencyHealth {
        case .ok: return .primary
        case .warn: return .yellow
        case .critical: return .red
        case .unknown: return .secondary
        }
    }

    /// Scrollable timeline of session events. Collapsed by default; stays
    /// visible after Stop so the user can inspect what happened.
    private var eventLogCard: some View {
        DisclosureGroup(isExpanded: $eventLogExpanded) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(bridge.realtimeEvents.reversed()) { ev in
                        HStack(alignment: .top, spacing: 6) {
                            Text(Self.logTimeFormatter.string(from: ev.timestamp))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(width: 90, alignment: .leading)
                            Text(ev.kind)
                                .font(.caption2).bold()
                                .foregroundStyle(levelColor(ev.level))
                                .frame(width: 110, alignment: .leading)
                            Text(ev.message)
                                .font(.caption2)
                                .foregroundStyle(.primary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .padding(.vertical, 1)
                    }
                }
                .padding(6)
            }
            .frame(maxHeight: 180)
            .background(.secondary.opacity(0.08), in: .rect(cornerRadius: 6))
        } label: {
            HStack {
                Text("イベントログ (\(bridge.realtimeEvents.count))")
                    .font(.subheadline).bold()
                Spacer()
                if !bridge.realtimeEvents.isEmpty {
                    Button("クリア") { bridge.resetRealtimeDiagnostics() }
                        .buttonStyle(.borderless).font(.caption)
                }
            }
        }
        .padding(10)
        .background(.thinMaterial, in: .rect(cornerRadius: 8))
    }

    private func levelColor(_ l: String) -> Color {
        switch l {
        case "error": return .red
        case "warn":  return .orange
        default:      return .secondary
        }
    }

    // MARK: - RPC actions

    private func refreshDevices() async {
        do {
            let r: AudioDeviceListing = try await bridge.call(
                "list_audio_devices", params: JSONValue.object([:]))
            devices = r
        } catch {
            errorMsg = "デバイス取得失敗: \(error.localizedDescription)"
        }
    }

    private func start() async {
        errorMsg = nil
        bridge.resetRealtimeDiagnostics()
        bridge.appendRealtimeEvent(
            level: "info", kind: "client",
            message: "ユーザーが開始をリクエスト (model=\(selectedModel), f0=\(f0Method))")
        let modelsPath = (bridge.initialInfo?["paths"]?["weight_root"]?.stringValue ?? "")
        let pthPath = modelsPath.isEmpty ? selectedModel : modelsPath + "/" + selectedModel
        var params: [String: JSONValue] = [
            "pth_path": .string(pthPath),
            "index_path": .string(indexPath),
            "pitch": .number(pitch),
            "formant": .number(formant),
            "index_rate": .number(indexRate),
            "threshold": .number(threshold),
            "block_time": .number(blockTime),
            "sample_rate": .number(sampleRate),
            "f0_method": .string(f0Method),
            "protect": .number(protect),
        ]
        if inputDeviceIndex >= 0 { params["input_device"] = .number(Double(inputDeviceIndex)) }
        if outputDeviceIndex >= 0 { params["output_device"] = .number(Double(outputDeviceIndex)) }

        do {
            struct R: Decodable {
                let status: String
                let sample_rate: Int?
                let model_sr: Int?
                let error: String?
                let input_device_name: String?
                let output_device_name: String?
                let block_time_ms: Double?
            }
            let r: R = try await bridge.call(
                "realtime_start",
                params: JSONValue.object(params),
                timeout: 60)
            if r.status == "success" {
                isRunning = true
                var parts: [String] = []
                parts.append("稼働中")
                parts.append("出力 \(r.sample_rate ?? 0) Hz")
                parts.append("モデル \(r.model_sr ?? 0) Hz")
                if let name = r.input_device_name, !name.isEmpty {
                    parts.append("in: \(name)")
                }
                if let name = r.output_device_name, !name.isEmpty {
                    parts.append("out: \(name)")
                }
                lastInfo = parts.joined(separator: " — ")
            } else {
                errorMsg = r.error ?? "開始失敗"
                bridge.appendRealtimeEvent(
                    level: "error", kind: "client",
                    message: r.error ?? "開始失敗 (不明)")
            }
        } catch {
            errorMsg = error.localizedDescription
            bridge.appendRealtimeEvent(
                level: "error", kind: "client", message: error.localizedDescription)
        }
    }

    private func stop() async {
        bridge.appendRealtimeEvent(
            level: "info", kind: "client", message: "ユーザーが停止をリクエスト")
        // Set isRunning=false BEFORE awaiting realtime_stop. The backend emits
        // its "status":"idle" notification before the RPC response, so it would
        // arrive during this await and drive realtimeRunning→false; the
        // .onChange handler must see isRunning already false (a no-op) rather
        // than treating a clean stop as a fatal error and surfacing a stale
        // lastError. Also gives immediate UI feedback.
        isRunning = false
        lastInfo = ""
        do {
            _ = try await bridge.callRaw(
                "realtime_stop",
                params: JSONValue.object([:]),
                timeout: 10)
        } catch {
            errorMsg = error.localizedDescription
        }
    }

    private func pushParamUpdate() async {
        let params: JSONValue = .object([
            "pitch": .number(pitch),
            "formant": .number(formant),
            "index_rate": .number(indexRate),
        ])
        _ = try? await bridge.callRaw(
            "realtime_update_params", params: params, timeout: 5)
    }
}
