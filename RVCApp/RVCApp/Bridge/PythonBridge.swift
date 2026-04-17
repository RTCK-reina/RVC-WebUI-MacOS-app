import Foundation
import Combine

/// Thrown when the Python subprocess cannot be launched or dies unexpectedly.
enum PythonBridgeError: Error, LocalizedError {
    case notLaunched
    case launchFailed(String)
    case processExited(Int32)
    case timeout
    case rpcError(JSONRPCError)
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .notLaunched: return "Python backend has not started yet."
        case .launchFailed(let s): return "Failed to launch Python backend: \(s)"
        case .processExited(let code):
            return "Python backend exited unexpectedly (code \(code))."
        case .timeout: return "Python backend did not respond in time."
        case .rpcError(let e): return "Backend error \(e.code): \(e.message)"
        case .invalidResponse(let s): return "Invalid backend response: \(s)"
        }
    }
}

/// Owns the Python subprocess, owns the stdin/stdout pipes, and multiplexes
/// concurrent RPC calls over a single pipe pair. `@MainActor` because every
/// published property drives SwiftUI directly.
@MainActor
final class PythonBridge: ObservableObject {
    // MARK: - Published state

    @Published private(set) var isReady: Bool = false
    @Published private(set) var isAlive: Bool = false
    @Published private(set) var resourceStats: ResourceStats = .idle
    @Published private(set) var activeProgress: [String: TaskProgress] = [:]
    /// Whether the floating task list is collapsed by the user.
    @Published var taskListMinimized: Bool = false
    /// Task IDs that have been cancelled — progress notifications for these are ignored.
    private var cancelledTaskIDs: Set<String> = []
    @Published private(set) var lastError: String?
    @Published private(set) var backendStatus: String = "starting"
    @Published private(set) var initialInfo: JSONValue?

    // Most recent model lists (updated by callers via list_models / list_indices).
    @Published var models: [String] = []
    @Published var indices: [String] = []
    @Published var uvr5Models: [String] = []

    // MARK: - Process plumbing

    /// PID cached for nonisolated killSync() — updated on start/shutdown.
    nonisolated(unsafe) private var _backendPID: pid_t = 0
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?

    private var nextId: Int = 1
    private var pendingRequests: [Int: CheckedContinuation<JSONValue, Error>] = [:]

    /// Accumulates partial stdout data — Python writes line-delimited JSON but
    /// pipe reads may split lines.
    private var stdoutBuffer = Data()

    /// Rolling buffer of recent stderr text from Python. Surfaced in the
    /// launch-error overlay so users can see what went wrong.
    private var stderrTail = ""
    private let stderrTailMaxLen = 8192
    /// Log file mirroring stderr for post-mortem debugging. Lives under
    /// ~/Library/Logs/RVC-Swift/.
    private var logFileHandle: FileHandle?

    // MARK: - Public lifecycle

    /// Launch the Python backend. Caller provides:
    ///  - `pythonExec`: absolute path to the bundled python3 binary
    ///  - `serverScript`: absolute path to rpc_server.py
    ///  - `baseDir`: read-only bundle dir (Contents/Resources)
    ///  - `userDir`: writable per-user dir (~/Documents/RVC-Swift by default)
    func start(
        pythonExec: String,
        serverScript: String,
        baseDir: String,
        userDir: String
    ) async throws {
        guard process == nil else {
            log("=== start() skipped — already running")
            return
        }

        openLogFile()
        log("=== start() called")
        log("  python:   \(pythonExec)")
        log("  script:   \(serverScript)")
        log("  base_dir: \(baseDir)")
        log("  user_dir: \(userDir)")

        let p = Process()
        p.executableURL = URL(fileURLWithPath: pythonExec)
        p.arguments = [
            serverScript,
            "--base-dir", baseDir,
            "--user-dir", userDir,
            "--nocheck",
        ]

        // Propagate a minimal, deterministic environment. We keep PATH in case
        // the python interpreter forks helper subprocesses.
        var env = ProcessInfo.processInfo.environment
        env["RVC_BASE_DIR"] = baseDir
        env["RVC_USER_DIR"] = userDir
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        // Belt-and-braces: some macOS sandbox contexts scrub HOME; make sure
        // Python can find a writable temp/cache location for conda-packed libs.
        if env["HOME"] == nil {
            env["HOME"] = NSHomeDirectory()
        }
        p.environment = env

        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe

        self.stdinPipe = inPipe
        self.stdoutPipe = outPipe
        self.stderrPipe = errPipe
        self.process = p

        // Install handlers BEFORE p.run() so no early output is lost.
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { return }
            Task { @MainActor [weak self] in
                self?.ingestStdout(data)
            }
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor [weak self] in
                self?.ingestStderr(data)
            }
        }

        // Detect process exit so we can surface it and fail pending calls.
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor [weak self] in
                self?.handleTermination(code: proc.terminationStatus)
            }
        }

        do {
            try p.run()
        } catch {
            throw PythonBridgeError.launchFailed(error.localizedDescription)
        }
        _backendPID = p.processIdentifier
        log("=== process.run() returned, pid=\(p.processIdentifier)")

        // Two-stage wait: first see an "alive" notification (fast, pre-import),
        // then wait for "ready" which fires after heavy imports complete.
        do {
            try await waitForAlive(timeout: 20)
            log("=== got alive")
        } catch {
            log("=== never saw alive: \(error.localizedDescription)")
            throw enrichLaunchError(error)
        }

        do {
            try await waitForReady(timeout: 120)
            log("=== got ready")
        } catch {
            log("=== never saw ready: \(error.localizedDescription)")
            throw enrichLaunchError(error)
        }

        // Fetch initialization info.
        let info: JSONValue = try await call("initialize", params: JSONValue.object([:]))
        self.initialInfo = info
    }

    /// Wrap a timeout error with the tail of the Python stderr so users can
    /// actually see what went wrong.
    private func enrichLaunchError(_ error: Error) -> Error {
        if case PythonBridgeError.timeout = error {
            let tail = String(stderrTail.suffix(1500))
            return PythonBridgeError.launchFailed(
                "Python backend did not respond in time.\n" +
                "ログ: ~/Library/Logs/RVC-Swift/bridge.log\n\n" +
                "--- Python stderr (末尾) ---\n\(tail)")
        }
        return error
    }

    /// Gracefully shutdown the backend.
    func shutdown() async {
        guard let p = process else { return }
        do {
            _ = try await callRaw("shutdown", params: JSONValue.object([:]), timeout: 3)
        } catch {
            // Ignore — we're going to kill it anyway if it doesn't exit.
        }
        // Give it a moment to exit on its own, then terminate.
        try? await Task.sleep(nanoseconds: 500_000_000)
        if p.isRunning {
            p.terminate()
        }
        process = nil
        _backendPID = 0
    }

    /// Remove a task and its children (prefix match) from the active progress
    /// list and ignore future updates for them.
    func clearProgress(for taskID: String) {
        // Snapshot keys to avoid mutating dictionary during iteration.
        let keysToRemove = activeProgress.keys.filter {
            $0 == taskID || $0.hasPrefix(taskID + "_")
        }
        for key in keysToRemove {
            activeProgress.removeValue(forKey: key)
            cancelledTaskIDs.insert(key)
        }
        cancelledTaskIDs.insert(taskID)
    }

    /// Synchronous kill for use in applicationWillTerminate where async
    /// is not available. Sends SIGTERM then SIGKILL to ensure cleanup.
    /// Uses the cached PID so it can be called from any isolation context.
    nonisolated func killSync() {
        let pid = _backendPID
        guard pid > 0 else { return }
        kill(pid, SIGTERM)
        usleep(300_000) // 300ms
        // Force-kill if still alive.
        if kill(pid, 0) == 0 {
            kill(pid, SIGKILL)
        }
    }

    // MARK: - RPC call helpers

    /// Typed call: params are any Encodable, response decoded as T.
    func call<T: Decodable>(
        _ method: String,
        params: some Encodable,
        timeout: TimeInterval = 300
    ) async throws -> T {
        let paramsValue = try paramsToJSONValue(params)
        let result = try await callRaw(method, params: paramsValue, timeout: timeout)
        // Round-trip through JSONEncoder/Decoder to get a typed T.
        let data = try JSONEncoder().encode(result)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw PythonBridgeError.invalidResponse(
                "Could not decode \(T.self) from \(String(data: data, encoding: .utf8) ?? "<?>"): \(error)"
            )
        }
    }

    /// Raw call: returns the JSONValue result exactly as the backend sent it.
    func callRaw(
        _ method: String,
        params: JSONValue,
        timeout: TimeInterval = 300
    ) async throws -> JSONValue {
        guard process != nil, let stdin = stdinPipe else {
            throw PythonBridgeError.notLaunched
        }
        let id = nextId
        nextId += 1

        let req = JSONRPCRawRequest(id: id, method: method, params: params)
        let encoded = try JSONEncoder().encode(req)
        var payload = encoded
        payload.append(0x0A)  // newline

        return try await withThrowingTaskGroup(of: JSONValue.self) { group in
            group.addTask { [weak self] in
                try await withCheckedThrowingContinuation { (cont: CheckedContinuation<JSONValue, Error>) in
                    Task { @MainActor [weak self] in
                        self?.pendingRequests[id] = cont
                    }
                    try? stdin.fileHandleForWriting.write(contentsOf: payload)
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw PythonBridgeError.timeout
            }
            guard let result = try await group.next() else {
                throw PythonBridgeError.invalidResponse("No response task produced a value")
            }
            group.cancelAll()
            return result
        }
    }

    // MARK: - Stdout parsing

    private func ingestStdout(_ data: Data) {
        stdoutBuffer.append(data)
        // Process each complete line (LF-terminated).
        while let nl = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = stdoutBuffer.subdata(in: stdoutBuffer.startIndex..<nl)
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...nl)
            if lineData.isEmpty { continue }
            handleLine(lineData)
        }
    }

    private func handleLine(_ data: Data) {
        do {
            let env = try JSONDecoder().decode(JSONRPCEnvelope.self, from: data)
            if let id = env.id {
                resolvePending(id: id, env: env)
            } else if let method = env.method {
                handleNotification(method: method, params: env.params ?? .null)
            }
        } catch {
            if let s = String(data: data, encoding: .utf8) {
                FileHandle.standardError.write(
                    Data("[py-parse-err] \(s)\n".utf8))
            }
        }
    }

    private func resolvePending(id: Int, env: JSONRPCEnvelope) {
        guard let cont = pendingRequests.removeValue(forKey: id) else { return }
        if let err = env.error {
            cont.resume(throwing: PythonBridgeError.rpcError(err))
        } else if let result = env.result {
            cont.resume(returning: result)
        } else {
            cont.resume(returning: .null)
        }
    }

    /// Stderr ingress: mirror to Swift stderr, accumulate tail, write to log file.
    private func ingestStderr(_ data: Data) {
        guard let s = String(data: data, encoding: .utf8) else { return }
        stderrTail += s
        if stderrTail.count > stderrTailMaxLen {
            stderrTail.removeFirst(stderrTail.count - stderrTailMaxLen)
        }
        FileHandle.standardError.write(
            ("[py] " + s).data(using: .utf8) ?? Data())
        try? logFileHandle?.write(contentsOf: s.data(using: .utf8) ?? Data())
    }

    private func handleNotification(method: String, params: JSONValue) {
        switch method {
        case "alive":
            self.isAlive = true
            self.backendStatus = "booting"
        case "ready":
            self.isAlive = true
            self.isReady = true
        case "resource_stats":
            self.resourceStats = ResourceStats(fromParams: params)
            self.backendStatus = self.resourceStats.status
        case "status":
            if let s = params["status"]?.stringValue {
                self.backendStatus = s
            }
        case "progress":
            let p = TaskProgress(fromParams: params)
            // Ignore updates for tasks that have been cancelled/cleared
            // (exact match or parent prefix match).
            let dominated = cancelledTaskIDs.contains(p.id)
                || cancelledTaskIDs.contains(where: { p.id.hasPrefix($0 + "_") })
            guard !dominated else { break }
            if var existing = self.activeProgress[p.id] {
                existing.update(with: params)
                self.activeProgress[p.id] = existing
            } else {
                self.activeProgress[p.id] = p
            }
            // Auto-clear completed entries after a short delay so the UI can
            // briefly show the 100% state.
            if p.percent >= 100 {
                let id = p.id
                cancelledTaskIDs.insert(id)
                Task { @MainActor [weak self] in
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    self?.activeProgress.removeValue(forKey: id)
                }
            }
        case "shutting_down":
            self.backendStatus = "shutting_down"
        case "realtime_error":
            if let msg = params["error"]?.stringValue {
                self.lastError = "リアルタイムVCエラー: \(msg)"
            }
        default:
            break
        }
    }

    // MARK: - Termination + readiness

    private func handleTermination(code: Int32) {
        self.isReady = false
        self.lastError = "Python backend exited with code \(code)"
        // Fail any pending requests.
        let pending = pendingRequests
        pendingRequests.removeAll()
        for (_, cont) in pending {
            cont.resume(throwing: PythonBridgeError.processExited(code))
        }
    }

    private func waitForReady(timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if isReady { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw PythonBridgeError.timeout
    }

    private func waitForAlive(timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if isAlive { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw PythonBridgeError.timeout
    }

    // MARK: - Logging

    private func openLogFile() {
        let fm = FileManager.default
        let libLogs = fm.urls(for: .libraryDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Logs/RVC-Swift", isDirectory: true)
        guard let dir = libLogs else { return }
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("bridge.log")
        if !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil)
        }
        if let h = try? FileHandle(forWritingTo: url) {
            try? h.seekToEnd()
            logFileHandle = h
            let ts = ISO8601DateFormatter().string(from: Date())
            try? h.write(contentsOf: Data("\n--- \(ts) ---\n".utf8))
        }
    }

    private func log(_ s: String) {
        let line = s + "\n"
        FileHandle.standardError.write(Data(line.utf8))
        try? logFileHandle?.write(contentsOf: Data(line.utf8))
    }

    // MARK: - Helpers

    private func paramsToJSONValue(_ value: some Encodable) throws -> JSONValue {
        let data = try JSONEncoder().encode(value)
        return try JSONDecoder().decode(JSONValue.self, from: data)
    }
}

// Dictionary helper to build JSONValue params inline.
extension JSONValue {
    static func obj(_ dict: [String: JSONValue]) -> JSONValue {
        .object(dict)
    }
}
