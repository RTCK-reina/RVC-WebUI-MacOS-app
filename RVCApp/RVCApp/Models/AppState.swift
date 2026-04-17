import Foundation
import SwiftUI

/// App-wide navigation + lifecycle state.
@MainActor
final class AppState: ObservableObject {
    @Published var selection: NavigationItem = .inferenceSingle
    @Published var launchError: String?
    @Published var isLaunching: Bool = true

    /// Launch the Python backend using the paths the .app bundle provides,
    /// falling back to dev-tree paths so the Swift app is runnable from Xcode
    /// without a full bundle build.
    private var didBootstrap = false

    func bootstrap(bridge: PythonBridge) async {
        guard !didBootstrap else { return }
        didBootstrap = true
        isLaunching = true
        defer { isLaunching = false }

        let (pythonExec, serverScript, baseDir) = resolveLaunchPaths()
        let userDir = resolveUserDir()

        // Make sure the user directory tree exists before Python runs.
        try? ensureUserDir(userDir)

        do {
            try await bridge.start(
                pythonExec: pythonExec,
                serverScript: serverScript,
                baseDir: baseDir,
                userDir: userDir)
            // Kick off a models/indices refresh immediately.
            async let models: [String: [String]] = bridge.call(
                "list_models", params: JSONValue.object([:]))
            async let indices: [String: [String]] = bridge.call(
                "list_indices", params: JSONValue.object([:]))
            async let uvr5: [String: [String]] = bridge.call(
                "list_uvr5_models", params: JSONValue.object([:]))
            if let m = try? await models["models"] { bridge.models = m }
            if let i = try? await indices["indices"] { bridge.indices = i }
            if let u = try? await uvr5["models"] { bridge.uvr5Models = u }
        } catch {
            self.launchError = error.localizedDescription
        }
    }

    /// Determine which python3 binary and rpc_server.py to launch.
    ///
    /// Priority:
    ///   1. Bundled resources at .../Contents/Resources/python/bin/python3
    ///   2. Development fallback: project root next to the Xcode project
    private func resolveLaunchPaths() -> (python: String, script: String, baseDir: String) {
        // Bundled layout.
        if let res = Bundle.main.resourcePath {
            let bundledPython = res + "/python/bin/python3"
            let bundledScript = res + "/rvc_backend/rpc_server.py"
            if FileManager.default.isExecutableFile(atPath: bundledPython),
               FileManager.default.fileExists(atPath: bundledScript) {
                return (bundledPython, bundledScript, res + "/rvc_backend")
            }
        }
        // Development fallback: walk up from the Xcode-built binary until we
        // find rpc_server.py alongside configs/.
        let fm = FileManager.default
        var dir = URL(fileURLWithPath: CommandLine.arguments[0])
            .deletingLastPathComponent()
        for _ in 0..<12 {
            let candidate = dir.appendingPathComponent("rpc_server.py")
            if fm.fileExists(atPath: candidate.path) {
                return (resolveDevPython(), candidate.path, dir.path)
            }
            dir.deleteLastPathComponent()
        }
        // Last-ditch: use cwd.
        return (
            resolveDevPython(),
            FileManager.default.currentDirectoryPath + "/rpc_server.py",
            FileManager.default.currentDirectoryPath
        )
    }

    /// Resolve a Python 3 interpreter path for the development fallback.
    ///
    /// Returns an absolute, executable path. Process.executableURL requires
    /// a real file path — passing a string like "/usr/bin/env python3" as
    /// a single executable would fail with ENOENT.
    ///
    /// Priority:
    ///   1. RVC_PYTHON env var (must point at an executable file)
    ///   2. Well-known macOS interpreter locations
    ///   3. Last-resort fallback to /usr/bin/python3 (a common location on
    ///      macOS, though not guaranteed; if absent the launcher will surface
    ///      a real ENOENT rather than the misleading literal "env python3" path)
    private func resolveDevPython() -> String {
        let fm = FileManager.default

        if let envPython = ProcessInfo.processInfo.environment["RVC_PYTHON"],
           !envPython.isEmpty,
           fm.isExecutableFile(atPath: envPython) {
            return envPython
        }

        let candidates = [
            "/opt/homebrew/Caskroom/miniforge/base/envs/rvc/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        for path in candidates {
            if fm.isExecutableFile(atPath: path) {
                return path
            }
        }
        return "/usr/bin/python3"
    }

    private func resolveUserDir() -> String {
        if let env = ProcessInfo.processInfo.environment["RVC_USER_DIR"] {
            return (env as NSString).expandingTildeInPath
        }
        let docs = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask
        ).first
        let root = docs ?? URL(fileURLWithPath: NSHomeDirectory())
        return root.appendingPathComponent("RVC-Swift").path
    }

    private func ensureUserDir(_ path: String) throws {
        let fm = FileManager.default
        let subdirs = [
            "input/audio",
            "input/training",
            "output/inference",
            "output/batch",
            "output/separation/vocals",
            "output/separation/accompaniment",
            "output/onnx",
            "models",
            "indices",
            "logs",
            "configs/inuse",
            "temp",
        ]
        for sd in subdirs {
            let target = (path as NSString).appendingPathComponent(sd)
            try fm.createDirectory(
                atPath: target, withIntermediateDirectories: true)
        }
    }
}

enum NavigationItem: Hashable, Identifiable {
    case inferenceSingle
    case inferenceBatch
    case separation
    case training
    case modelsCompare
    case modelsMerge
    case modelsInfo
    case modelsExtract
    case onnxExport
    case realtimeVC

    var id: Int { hashValue }

    var title: String {
        switch self {
        case .inferenceSingle: return "単一推論"
        case .inferenceBatch:  return "バッチ推論"
        case .separation:      return "UVR5 分離"
        case .training:        return "トレーニング"
        case .modelsCompare:   return "モデル比較"
        case .modelsMerge:     return "モデル融合"
        case .modelsInfo:      return "モデル情報"
        case .modelsExtract:   return "モデル抽出"
        case .onnxExport:      return "ONNX 出力"
        case .realtimeVC:      return "リアルタイム VC"
        }
    }

    var systemImage: String {
        switch self {
        case .inferenceSingle: return "waveform"
        case .inferenceBatch:  return "square.stack.3d.up"
        case .separation:      return "scissors"
        case .training:        return "cpu"
        case .modelsCompare:   return "equal.square"
        case .modelsMerge:     return "arrow.triangle.merge"
        case .modelsInfo:      return "info.circle"
        case .modelsExtract:   return "arrow.down.circle"
        case .onnxExport:      return "arrow.right.doc"
        case .realtimeVC:      return "mic"
        }
    }
}
