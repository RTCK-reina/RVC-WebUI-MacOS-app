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

    private let f0Methods = ["pm", "harvest", "dio", "crepe", "rmvpe", "fcpe"]

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
            }
            let r: R = try await bridge.call(
                "realtime_start",
                params: JSONValue.object(params),
                timeout: 60)
            if r.status == "success" {
                isRunning = true
                lastInfo = "稼働中 — 出力 \(r.sample_rate ?? 0) Hz / モデル \(r.model_sr ?? 0) Hz"
            } else {
                errorMsg = r.error ?? "開始失敗"
            }
        } catch {
            errorMsg = error.localizedDescription
        }
    }

    private func stop() async {
        do {
            _ = try await bridge.callRaw(
                "realtime_stop",
                params: JSONValue.object([:]),
                timeout: 10)
        } catch {
            errorMsg = error.localizedDescription
        }
        isRunning = false
        lastInfo = ""
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
