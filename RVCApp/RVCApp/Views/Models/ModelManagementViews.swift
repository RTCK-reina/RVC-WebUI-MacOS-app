import SwiftUI

// MARK: - Compare

struct ModelCompareView: View {
    @EnvironmentObject var bridge: PythonBridge
    @State private var idA = ""
    @State private var idB = ""
    @State private var result: String?
    @State private var error: String?

    var body: some View {
        Form {
            Section("2つの ID-long (hash) を比較") {
                TextField("ID A", text: $idA)
                TextField("ID B", text: $idB)
                Button("比較") { Task { await run() } }
            }
            if let r = result {
                Section("結果") {
                    Text("類似度: \(r)").font(.system(.body, design: .monospaced))
                }
            }
            if let err = error {
                Label(err, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func run() async {
        error = nil
        result = nil
        do {
            struct R: Decodable { let similarity: String }
            let r: R = try await bridge.call("model_compare",
                params: JSONValue.object([
                    "id_a": .string(idA), "id_b": .string(idB)
                ]))
            result = r.similarity
        } catch {
            self.error = error.localizedDescription
        }
    }
}

// MARK: - Merge

struct ModelMergeView: View {
    @EnvironmentObject var bridge: PythonBridge
    @State private var pathA = ""
    @State private var pathB = ""
    @State private var alpha: Double = 0.5
    @State private var sr: Int = 40000
    @State private var ifF0: Bool = true
    @State private var name = ""
    @State private var info = ""
    @State private var version = "v2"
    @State private var resultMsg: String?
    @State private var errorMsg: String?
    @State private var taskID = ""
    @State private var isRunning = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("モデル融合").font(.title2).bold()
                FilePickerField(label: "モデル A (.pth)", path: $pathA,
                    allowedContentTypes: [], chooseDirectory: false)
                FilePickerField(label: "モデル B (.pth)", path: $pathB,
                    allowedContentTypes: [], chooseDirectory: false)
                HStack {
                    Text("α").frame(width: 40)
                    Slider(value: $alpha, in: 0...1, step: 0.01)
                    Text(String(format: "%.2f", alpha))
                        .font(.caption.monospacedDigit()).frame(width: 40)
                }
                Picker("サンプリングレート", selection: $sr) {
                    Text("32000").tag(32000)
                    Text("40000").tag(40000)
                    Text("48000").tag(48000)
                }
                Toggle("F0 あり", isOn: $ifF0)
                Picker("Version", selection: $version) {
                    Text("v1").tag("v1"); Text("v2").tag("v2")
                }
                TextField("新モデル名", text: $name)
                TextField("備考", text: $info)
                Button("融合実行") { Task { await run() } }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning || pathA.isEmpty || pathB.isEmpty || name.isEmpty)
                if !taskID.isEmpty {
                    // Model merge is a single torch.save call — cannot be
                    // interrupted mid-flight.
                    ProgressBarView(taskID: taskID, title: "融合中")
                }
                if let r = resultMsg {
                    Text(r).font(.caption.monospacedDigit())
                }
                if let e = errorMsg {
                    Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                }
            }
            .padding()
        }
    }

    private func run() async {
        resultMsg = nil; errorMsg = nil
        let id = "model_merge_\(Int(Date().timeIntervalSince1970 * 1000))"
        taskID = id
        isRunning = true
        defer { isRunning = false }
        do {
            struct R: Decodable { let result: String? }
            let r: R = try await bridge.call("model_merge",
                params: JSONValue.object([
                    "task_id": .string(id),
                    "path_a": .string(pathA),
                    "path_b": .string(pathB),
                    "alpha": .number(alpha),
                    "sr": .number(Double(sr)),
                    "if_f0": .number(ifF0 ? 1 : 0),
                    "info": .string(info),
                    "name": .string(name),
                    "version": .string(version),
                ]), timeout: 3600)
            resultMsg = r.result
        } catch { errorMsg = error.localizedDescription }
    }
}

// MARK: - Info

struct ModelInfoView: View {
    @EnvironmentObject var bridge: PythonBridge
    @State private var path = ""
    @State private var info: String = ""
    @State private var errorMsg: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("モデル情報").font(.title2).bold()
                FilePickerField(label: "モデルファイル (.pth)", path: $path,
                    allowedContentTypes: [], chooseDirectory: false)
                Button("情報取得") { Task { await run() } }
                    .disabled(path.isEmpty)
                if !info.isEmpty {
                    Text(info)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.thinMaterial, in: .rect(cornerRadius: 6))
                }
                if let e = errorMsg {
                    Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                }
            }
            .padding()
        }
    }

    private func run() async {
        errorMsg = nil
        do {
            struct R: Decodable { let info: String }
            let r: R = try await bridge.call("model_info",
                params: JSONValue.object(["path": .string(path)]))
            info = r.info
        } catch { errorMsg = error.localizedDescription }
    }
}

// MARK: - Extract

struct ModelExtractView: View {
    @EnvironmentObject var bridge: PythonBridge
    @State private var ckpt = ""
    @State private var name = ""
    @State private var info = ""
    @State private var sr: Int = 40000
    @State private var ifF0 = true
    @State private var version = "v2"
    @State private var msg: String?
    @State private var errorMsg: String?
    @State private var taskID = ""
    @State private var isRunning = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("モデル抽出 (軽量化)").font(.title2).bold()
                FilePickerField(label: "チェックポイント (.pth)",
                    path: $ckpt, allowedContentTypes: [])
                TextField("新モデル名", text: $name)
                TextField("備考", text: $info)
                Picker("サンプリングレート", selection: $sr) {
                    Text("32000").tag(32000); Text("40000").tag(40000); Text("48000").tag(48000)
                }
                Toggle("F0 あり", isOn: $ifF0)
                Picker("Version", selection: $version) {
                    Text("v1").tag("v1"); Text("v2").tag("v2")
                }
                Button("抽出") { Task { await run() } }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning || ckpt.isEmpty || name.isEmpty)
                if !taskID.isEmpty {
                    // Extract is a single ckpt save — cannot be interrupted.
                    ProgressBarView(taskID: taskID, title: "抽出中")
                }
                if let m = msg { Text(m).font(.caption.monospacedDigit()) }
                if let e = errorMsg {
                    Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                }
            }
            .padding()
        }
    }

    private func run() async {
        msg = nil; errorMsg = nil
        let id = "model_extract_\(Int(Date().timeIntervalSince1970 * 1000))"
        taskID = id
        isRunning = true
        defer { isRunning = false }
        do {
            struct R: Decodable { let result: String? }
            let r: R = try await bridge.call("model_extract",
                params: JSONValue.object([
                    "task_id": .string(id),
                    "ckpt_path": .string(ckpt),
                    "name": .string(name),
                    "sr": .number(Double(sr)),
                    "if_f0": .number(ifF0 ? 1 : 0),
                    "info": .string(info),
                    "version": .string(version),
                ]), timeout: 3600)
            msg = r.result
        } catch { errorMsg = error.localizedDescription }
    }
}

// MARK: - ONNX Export

struct OnnxExportView: View {
    @EnvironmentObject var bridge: PythonBridge
    @State private var ckpt = ""
    @State private var output = ""
    @State private var msg: String?
    @State private var errorMsg: String?
    @State private var taskID = ""
    @State private var isRunning = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("ONNX エクスポート").font(.title2).bold()
                FilePickerField(label: "チェックポイント (.pth)",
                    path: $ckpt, allowedContentTypes: [])
                FilePickerField(label: "出力先 (.onnx)",
                    path: $output, allowedContentTypes: [],
                    initialDir: defaultOutputDir)
                Button("エクスポート") { Task { await run() } }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning || ckpt.isEmpty)
                if !taskID.isEmpty {
                    // ONNX export is a single torch.onnx.export call.
                    ProgressBarView(taskID: taskID, title: "ONNX 出力中")
                }
                if let m = msg { Text(m).font(.caption.monospacedDigit()) }
                if let e = errorMsg {
                    Label(e, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                }
            }
            .padding()
        }
    }

    private var defaultOutputDir: String? {
        (bridge.initialInfo?["paths"]?["output_root"]?.stringValue).flatMap { $0 + "/onnx" }
    }

    private func run() async {
        msg = nil; errorMsg = nil
        let id = "export_onnx_\(Int(Date().timeIntervalSince1970 * 1000))"
        taskID = id
        isRunning = true
        defer { isRunning = false }
        do {
            struct R: Decodable {
                let status: String
                let output_path: String?
            }
            let r: R = try await bridge.call("export_onnx",
                params: JSONValue.object([
                    "task_id": .string(id),
                    "ckpt_path": .string(ckpt),
                    "output_path": .string(output),
                ]), timeout: 3600)
            msg = r.output_path ?? r.status
        } catch { errorMsg = error.localizedDescription }
    }
}
