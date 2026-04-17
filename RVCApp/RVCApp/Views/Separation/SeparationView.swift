import SwiftUI

struct UVRResult: Decodable {
    let status: String
    let messages: [String]?
    let output_vocal: String?
    let output_accompaniment: String?
    let polished_dir: String?
}

/// Static knowledge base for the bundled UVR5 models.
enum UVR5ModelInfo {
    struct Details {
        let shortName: String     // Display name (concise)
        let purpose: String       // What it does (one line)
        let whenToUse: String     // When the user should pick it
        let strength: String      // Separation characteristics
        let category: Category    // For grouping
    }

    enum Category: String {
        case vocalSeparation = "ボーカル/伴奏分離"
        case mainVocalOnly   = "メインボーカル抽出"
        case deEcho          = "エコー・残響除去"
    }

    /// Returns details for a model file name (with or without the `.pth`).
    static func details(for modelName: String) -> Details? {
        let key = modelName.replacingOccurrences(of: ".pth", with: "")
        return table[key]
    }

    static let orderedKeys: [String] = [
        "HP2-人声vocals+非人声instrumentals",
        "HP2_all_vocals",
        "HP3_all_vocals",
        "HP5-主旋律人声vocals+其他instrumentals",
        "HP5_only_main_vocal",
        "VR-DeEchoNormal",
        "VR-DeEchoAggressive",
        "VR-DeEchoDeReverb",
        "onnx_dereverb_By_FoxJoy",
    ]

    private static let table: [String: Details] = [
        "HP2-人声vocals+非人声instrumentals": Details(
            shortName: "HP2 (ボーカル＋伴奏)",
            purpose: "標準的なボーカルと伴奏の分離",
            whenToUse: "普通の楽曲から歌声とカラオケを作りたい時の第一候補。バランス重視。",
            strength: "中程度の分離力、汎用向き",
            category: .vocalSeparation),

        "HP2_all_vocals": Details(
            shortName: "HP2 (全ボーカル)",
            purpose: "メイン＋ハモリ＋コーラスを全てボーカル側に残す",
            whenToUse: "ハモリやバックコーラスも一緒に抽出したい時。コーラス全体をまとめて扱いたい場合。",
            strength: "ボーカル要素を欠落させない",
            category: .vocalSeparation),

        "HP3_all_vocals": Details(
            shortName: "HP3 (全ボーカル・改良)",
            purpose: "HP2 の改良版。より高精度に全ボーカルを分離",
            whenToUse: "HP2 の全ボーカル版で物足りない時の上位互換。基本こちらを推奨。",
            strength: "HP2 より分離品質が高い",
            category: .vocalSeparation),

        "HP5-主旋律人声vocals+其他instrumentals": Details(
            shortName: "HP5 (主旋律のみ)",
            purpose: "リードボーカルのみ分離、ハモリは伴奏側に回す",
            whenToUse: "リード歌唱のみ抽出したい時。ハモリやコーラスは邪魔なので除外したい。",
            strength: "メインメロディに特化",
            category: .mainVocalOnly),

        "HP5_only_main_vocal": Details(
            shortName: "HP5 (メインボーカル専用)",
            purpose: "HP5 よりさらに厳しくメインボーカルのみ抽出",
            whenToUse: "ソロ歌唱をきれいに取り出したい時の最終候補。RVC 学習用の教師データ準備にも。",
            strength: "ハモリ・コーラスを強力に排除",
            category: .mainVocalOnly),

        "VR-DeEchoNormal": Details(
            shortName: "DeEcho (通常)",
            purpose: "控えめなエコー除去",
            whenToUse: "軽いエコーや空気感を残したい時。スタジオ録音のナチュラル処理向け。",
            strength: "自然な残響を保持しつつエコーを軽減",
            category: .deEcho),

        "VR-DeEchoAggressive": Details(
            shortName: "DeEcho (強)",
            purpose: "積極的にエコー成分を除去",
            whenToUse: "ライブ音源・ホール録音など強い残響を消したい時。ドライな音声が欲しい時。",
            strength: "エコーを強力除去、場合により音色も変化",
            category: .deEcho),

        "VR-DeEchoDeReverb": Details(
            shortName: "DeEcho + DeReverb",
            purpose: "エコーとリバーブを同時除去",
            whenToUse: "エコーとリバーブが両方かかっている音源。カラオケ・配信音声など。",
            strength: "2種類のウェット成分を一括処理",
            category: .deEcho),

        "onnx_dereverb_By_FoxJoy": Details(
            shortName: "ONNX DeReverb (FoxJoy)",
            purpose: "ONNX ベースの高品質リバーブ除去",
            whenToUse: "リバーブのみ除去したい時。ボーカル抽出後の仕上げに最適。",
            strength: "最新の ONNX 推論、軽量高品質",
            category: .deEcho),
    ]
}

struct SeparationView: View {
    @EnvironmentObject var bridge: PythonBridge

    @AppStorage("sep.modelName") private var modelName: String = ""
    @AppStorage("sep.dirPath") private var dirPath: String = ""
    @State private var paths: [String] = []  // session-specific file list
    @AppStorage("sep.outputVocal") private var outputVocal: String = ""
    @AppStorage("sep.outputIns") private var outputIns: String = ""
    @AppStorage("sep.agg") private var agg: Double = 10
    @AppStorage("sep.format") private var format: String = "flac"

    // Second-pass polish (dereverb / de-echo applied to the extracted vocals).
    @AppStorage("sep.polishEnabled") private var polishEnabled: Bool = false
    @AppStorage("sep.polishModel") private var polishModel: String = "onnx_dereverb_By_FoxJoy"

    @State private var taskID = ""
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var lastMessages: [String] = []
    @State private var lastVocalDir: String?
    @State private var lastInsDir: String?
    @State private var lastPolishedDir: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("UVR5 音声分離").font(.title2).bold()
                Text("ボーカルと伴奏を分離します。")
                    .font(.caption).foregroundStyle(.secondary)

                modelGuideCard

                GroupBox("モデル・入出力") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("モデル").font(.caption).foregroundStyle(.secondary)
                                .frame(width: 100, alignment: .leading)
                            Picker("", selection: $modelName) {
                                Text("（未選択）").tag("")
                                ForEach(groupedModels, id: \.0) { group, items in
                                    Section(header: Text(group)) {
                                        ForEach(items, id: \.self) { name in
                                            let details = UVR5ModelInfo.details(for: name)
                                            Text(details?.shortName ?? name).tag(name)
                                        }
                                    }
                                }
                            }
                            .labelsHidden()
                        }

                        if let details = UVR5ModelInfo.details(for: modelName) {
                            modelDetailCard(details)
                        }
                        FilePickerField(
                            label: "入力ディレクトリ",
                            path: $dirPath, chooseDirectory: true,
                            initialDir: defaultInputDir)
                        MultiFilePickerField(
                            label: "または個別ファイル",
                            paths: $paths,
                            initialDir: defaultInputDir)
                        FilePickerField(
                            label: "ボーカル出力先",
                            path: $outputVocal, chooseDirectory: true,
                            initialDir: defaultVocalDir)
                        FilePickerField(
                            label: "伴奏出力先",
                            path: $outputIns, chooseDirectory: true,
                            initialDir: defaultInsDir)
                        HStack {
                            Text("Aggression").font(.caption).foregroundStyle(.secondary)
                                .frame(width: 120, alignment: .leading)
                            Slider(value: $agg, in: 0...20, step: 1)
                            Text("\(Int(agg))").font(.caption.monospacedDigit()).frame(width: 40, alignment: .trailing)
                        }
                        Picker("出力形式", selection: $format) {
                            ForEach(["flac", "wav", "mp3", "m4a"], id: \.self) { Text($0.uppercased()).tag($0) }
                        }
                        .frame(width: 220)
                    }
                    .padding(8)
                }

                polishCard

                Button(action: { Task { await run() } }) {
                    Label(polishEnabled ? "分離 + 仕上げを実行" : "分離を実行",
                          systemImage: "scissors")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning
                    || modelName.isEmpty
                    || (dirPath.isEmpty && paths.isEmpty)
                    || (polishEnabled && polishModel.isEmpty))

                if !taskID.isEmpty {
                    ProgressBarView(taskID: taskID, title: "分離中", onCancel: cancel)
                }
                if let err = errorMessage {
                    Label(err, systemImage: "exclamationmark.triangle.fill")
                        .padding(10)
                        .foregroundStyle(.red)
                        .background(.red.opacity(0.1), in: .rect(cornerRadius: 8))
                }
                if lastVocalDir != nil || lastInsDir != nil || lastPolishedDir != nil {
                    GroupBox("保存先") {
                        VStack(alignment: .leading, spacing: 6) {
                            if let vocal = lastVocalDir {
                                outputRow(label: "ボーカル", path: vocal, icon: "music.mic")
                            }
                            if let ins = lastInsDir {
                                outputRow(label: "伴奏", path: ins, icon: "music.note.list")
                            }
                            if let polished = lastPolishedDir {
                                outputRow(label: "仕上げ済み", path: polished, icon: "sparkles")
                            }
                        }
                        .padding(8)
                    }
                }
                if !lastMessages.isEmpty {
                    GroupBox("ログ") {
                        ScrollView {
                            Text(lastMessages.joined(separator: "\n"))
                                .font(.caption.monospacedDigit())
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 160)
                    }
                }
            }
            .padding(16)
        }
    }

    // MARK: - Polish (auto chained second-pass) UI

    private var polishCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Toggle(isOn: $polishEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("分離後に仕上げを自動でかける").bold()
                        Text("抽出したボーカルをもう一度別モデルに通して、エコー・リバーブを除去します。2パス処理のため音質がわずかに劣化する場合があります。エコーやリバーブが強いソース向けです。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                if polishEnabled {
                    HStack {
                        Text("仕上げモデル")
                            .font(.caption).foregroundStyle(.secondary)
                            .frame(width: 100, alignment: .leading)
                        Picker("", selection: $polishModel) {
                            ForEach(polishCandidates, id: \.self) { name in
                                let d = UVR5ModelInfo.details(for: name)
                                Text(d?.shortName ?? name).tag(name)
                            }
                        }
                        .labelsHidden()
                    }
                    if let d = UVR5ModelInfo.details(for: polishModel) {
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "sparkles").foregroundStyle(.yellow)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(d.shortName).font(.caption).bold()
                                Text(d.purpose).font(.caption)
                                Text(d.whenToUse)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(8)
                        .background(.yellow.opacity(0.08), in: .rect(cornerRadius: 6))
                    }
                    Text("仕上げ結果は「ボーカル出力先/polished/」に書き出されます。")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            .padding(8)
        }
    }

    /// Only DeEcho / DeReverb category models are sensible polish passes.
    private var polishCandidates: [String] {
        let available = Set(bridge.uvr5Models)
        return UVR5ModelInfo.orderedKeys.filter { key in
            available.contains(key)
                && UVR5ModelInfo.details(for: key)?.category == .deEcho
        }
    }

    // MARK: - Model info UI

    /// Models grouped by UVR5ModelInfo.Category, preserving the canonical order.
    private var groupedModels: [(String, [String])] {
        let available = Set(bridge.uvr5Models)
        let canonical = UVR5ModelInfo.orderedKeys.filter { available.contains($0) }
        // Place any unknown models in an "その他" group at the end.
        let unknown = bridge.uvr5Models.filter { UVR5ModelInfo.details(for: $0) == nil }
        var grouped: [(String, [String])] = []
        for category in [UVR5ModelInfo.Category.vocalSeparation,
                         .mainVocalOnly,
                         .deEcho] {
            let items = canonical.filter {
                UVR5ModelInfo.details(for: $0)?.category == category
            }
            if !items.isEmpty {
                grouped.append((category.rawValue, items))
            }
        }
        if !unknown.isEmpty {
            grouped.append(("その他", unknown))
        }
        return grouped
    }

    private var modelGuideCard: some View {
        GroupBox("モデルの選び方") {
            VStack(alignment: .leading, spacing: 6) {
                guideRow(icon: "music.mic", title: "歌ってみた・カラオケ作成",
                         body: "HP3 (全ボーカル) か HP2 (ボーカル＋伴奏)。まず HP3 を試して物足りなければ HP2。")
                guideRow(icon: "person.wave.2", title: "リードボーカルのみ抽出",
                         body: "HP5 (メインボーカル専用)。RVC 学習用データ作成にも最適。")
                guideRow(icon: "waveform.path", title: "エコー・残響を取り除く",
                         body: "まず ONNX DeReverb、強い残響なら DeEcho (強)、両方なら DeEcho + DeReverb。")
                guideRow(icon: "arrow.triangle.branch", title: "ボーカル抽出後の仕上げ",
                         body: "エコー・リバーブが強い音源に有効。通常のスタジオ音源では仕上げなしの方が高品質な場合があります。")
            }
            .padding(8)
        }
    }

    private func guideRow(icon: String, title: String, body: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(.blue)
                .frame(width: 20)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.caption).bold()
                Text(body).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func modelDetailCard(_ d: UVR5ModelInfo.Details) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "info.circle.fill").foregroundStyle(.blue)
                Text(d.shortName).font(.subheadline).bold()
                Spacer()
                Text(d.category.rawValue)
                    .font(.caption2)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(.blue.opacity(0.15), in: .capsule)
            }
            labeledRow("用途", d.purpose)
            labeledRow("こんな時に", d.whenToUse)
            labeledRow("特徴", d.strength)
        }
        .padding(10)
        .background(.blue.opacity(0.07), in: .rect(cornerRadius: 8))
    }

    private func outputRow(label: String, path: String, icon: String) -> some View {
        HStack {
            Label(label, systemImage: icon)
                .font(.caption).bold()
                .frame(width: 100, alignment: .leading)
            Text(path)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button("Finderで開く") {
                NSWorkspace.shared.activateFileViewerSelecting(
                    [URL(fileURLWithPath: path)])
            }
            .font(.caption)
        }
    }

    private func labeledRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label)
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .frame(width: 72, alignment: .trailing)
            Text(value)
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var defaultInputDir: String? {
        (bridge.initialInfo?["paths"]?["input_root"]?.stringValue).flatMap { $0 + "/audio" }
    }

    private var defaultVocalDir: String? {
        (bridge.initialInfo?["paths"]?["output_root"]?.stringValue).flatMap {
            $0 + "/separation/vocals"
        }
    }

    private var defaultInsDir: String? {
        (bridge.initialInfo?["paths"]?["output_root"]?.stringValue).flatMap {
            $0 + "/separation/accompaniment"
        }
    }

    private func run() async {
        errorMessage = nil
        lastMessages = []
        lastVocalDir = nil
        lastInsDir = nil
        lastPolishedDir = nil
        let id = "uvr5_\(Int(Date().timeIntervalSince1970 * 1000))"
        taskID = id
        isRunning = true
        defer {
            isRunning = false
            taskID = ""
        }

        var paramsDict: [String: JSONValue] = [
            "task_id": .string(id),
            "model_name": .string(modelName),
            "input_dir": .string(dirPath),
            "paths": .array(paths.map { .string($0) }),
            "output_vocal": .string(outputVocal),
            "output_accompaniment": .string(outputIns),
            "agg": .number(agg),
            "format": .string(format),
        ]
        if polishEnabled && !polishModel.isEmpty {
            paramsDict["polish_model"] = .string(polishModel)
        }

        do {
            let r: UVRResult = try await bridge.call(
                "uvr5",
                params: JSONValue.object(paramsDict),
                timeout: 7200)
            lastMessages = r.messages ?? []
            lastVocalDir = r.output_vocal
            lastInsDir = r.output_accompaniment
            lastPolishedDir = r.polished_dir
            if r.status == "success" {
                AppNotification.send(
                    title: "音声分離が完了しました",
                    body: "ボーカルと伴奏の分離が正常に終了しました。")
            } else {
                errorMessage = "分離に失敗しました"
                AppNotification.send(
                    title: "音声分離に失敗しました",
                    body: "エラーが発生しました。ログを確認してください。")
            }
        } catch {
            errorMessage = error.localizedDescription
            AppNotification.send(
                title: "音声分離に失敗しました",
                body: error.localizedDescription)
        }
    }

    private func cancel() {
        let id = taskID
        Task {
            _ = try? await bridge.callRaw("cancel",
                params: .object(["task_id": .string(id)]))
            bridge.clearProgress(for: id)
        }
    }
}
