import Foundation

/// Per-block diagnostics for the realtime VC session, delivered ~10Hz via the
/// `realtime_metrics` JSON-RPC notification. Drives the meter, badge, and
/// numeric-table cards in `RealtimeVCView`.
struct RealtimeMetrics: Equatable {
    var timestamp: Date = .distantPast

    // Levels (dBFS, -90 floor, 0 ceiling).
    var inputRMSdB: Double = -90
    var outputRMSdB: Double = -90
    var inputPeakdB: Double = -90
    var outputPeakdB: Double = -90
    var inputClipped: Bool = false
    var outputClipped: Bool = false

    // True only when the noise-gate zeroed the whole block — lets the UI
    // distinguish "quiet because gated" from "mic returning zeros".
    var isGated: Bool = false

    // Latency / timing (ms).
    var inferenceMs: Double = 0
    var blockMs: Double = 0
    var blockBudgetMs: Double = 0

    // Cumulative counters over the session.
    var errorCount: Int = 0
    var inputUnderflow: Int = 0
    var inputOverflow: Int = 0
    var outputUnderflow: Int = 0
    var outputOverflow: Int = 0
    var primingOutput: Int = 0

    // Resolved device info (populated after stream.start()).
    var inputDeviceName: String = ""
    var outputDeviceName: String = ""
    var inputDeviceIndex: Int = -1
    var outputDeviceIndex: Int = -1
    var sampleRate: Int = 0
    var thresholdDB: Double = -60

    static let idle = RealtimeMetrics()

    enum LatencyHealth { case ok, warn, critical, unknown }

    /// How tight inference is against the real-time deadline. >1.0 means the
    /// inference couldn't keep up with the block — audio will glitch.
    var latencyHealth: LatencyHealth {
        guard blockBudgetMs > 0 else { return .unknown }
        let ratio = inferenceMs / blockBudgetMs
        if ratio < 0.7 { return .ok }
        if ratio < 1.0 { return .warn }
        return .critical
    }

    var totalUnderOverflow: Int {
        inputUnderflow + inputOverflow + outputUnderflow + outputOverflow
    }

    init() {}

    /// Parse from a `params` object of a `realtime_metrics` notification.
    init(fromParams json: JSONValue) {
        if let ts = json["timestamp"]?.doubleValue {
            self.timestamp = Date(timeIntervalSince1970: ts)
        }
        self.inputRMSdB   = json["input_rms_db"]?.doubleValue ?? -90
        self.outputRMSdB  = json["output_rms_db"]?.doubleValue ?? -90
        self.inputPeakdB  = json["input_peak_db"]?.doubleValue ?? -90
        self.outputPeakdB = json["output_peak_db"]?.doubleValue ?? -90
        self.inputClipped  = json["input_clipped"]?.boolValue ?? false
        self.outputClipped = json["output_clipped"]?.boolValue ?? false
        self.isGated       = json["is_gated"]?.boolValue ?? false

        self.inferenceMs   = json["inference_ms"]?.doubleValue ?? 0
        self.blockMs       = json["block_ms"]?.doubleValue ?? 0
        self.blockBudgetMs = json["block_time_budget_ms"]?.doubleValue ?? 0

        self.errorCount = json["error_count"]?.intValue ?? 0
        if let flags = json["callback_flags"] {
            self.inputUnderflow  = flags["input_underflow"]?.intValue ?? 0
            self.inputOverflow   = flags["input_overflow"]?.intValue ?? 0
            self.outputUnderflow = flags["output_underflow"]?.intValue ?? 0
            self.outputOverflow  = flags["output_overflow"]?.intValue ?? 0
            self.primingOutput   = flags["priming_output"]?.intValue ?? 0
        }

        self.inputDeviceIndex  = json["input_device_index"]?.intValue ?? -1
        self.outputDeviceIndex = json["output_device_index"]?.intValue ?? -1
        self.inputDeviceName   = json["input_device_name"]?.stringValue ?? ""
        self.outputDeviceName  = json["output_device_name"]?.stringValue ?? ""
        self.sampleRate        = json["samplerate"]?.intValue ?? 0
        self.thresholdDB       = json["threshold_db"]?.doubleValue ?? -60
    }
}

/// A single event in the realtime session's timeline — session start/stop,
/// parameter updates, callback errors, client-side button presses. Shown in
/// the collapsible log card.
struct RealtimeEvent: Identifiable, Equatable {
    let id: UUID
    let timestamp: Date
    let level: String   // "info" | "warn" | "error"
    let kind: String    // "started" | "stopped" | "callback_error" | "params_updated" | "client"
    let message: String

    init(timestamp: Date, level: String, kind: String, message: String) {
        self.id = UUID()
        self.timestamp = timestamp
        self.level = level
        self.kind = kind
        self.message = message
    }

    init(fromParams json: JSONValue) {
        let ts = json["timestamp"]?.doubleValue ?? Date().timeIntervalSince1970
        self.id = UUID()
        self.timestamp = Date(timeIntervalSince1970: ts)
        self.level = json["level"]?.stringValue ?? "info"
        self.kind = json["kind"]?.stringValue ?? "event"
        self.message = json["message"]?.stringValue ?? ""
    }
}
