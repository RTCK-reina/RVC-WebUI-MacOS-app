import Foundation

/// Snapshot of backend resource usage received via the `resource_stats`
/// notification (emitted ~once per second).
struct ResourceStats: Equatable {
    var timestamp: Date = .distantPast
    var status: String = "starting"
    var device: String = "unknown"
    var gpuBackend: String = "cpu"
    var isHalf: Bool = false

    // CPU + system memory.
    var cpuPercent: Double = 0
    var memoryUsedGB: Double = 0
    var memoryTotalGB: Double = 0
    var memoryPercent: Double = 0
    var processMemoryGB: Double = 0

    // GPU / accelerator.
    var gpuMemoryUsedMB: Int = 0
    var gpuMemoryReservedOrDriverMB: Int = 0

    static let idle = ResourceStats()

    init() {}

    /// Parse from a `params` object of a resource_stats notification.
    init(fromParams json: JSONValue) {
        if let ts = json["timestamp"]?.doubleValue {
            self.timestamp = Date(timeIntervalSince1970: ts)
        }
        self.status = json["status"]?.stringValue ?? "idle"
        self.device = json["device"]?.stringValue ?? "unknown"
        self.gpuBackend = json["gpu_backend"]?.stringValue ?? "cpu"
        self.isHalf = json["is_half"]?.boolValue ?? false

        self.cpuPercent = json["cpu_percent"]?.doubleValue ?? 0
        self.memoryUsedGB = json["memory_used_gb"]?.doubleValue ?? 0
        self.memoryTotalGB = json["memory_total_gb"]?.doubleValue ?? 0
        self.memoryPercent = json["memory_percent"]?.doubleValue ?? 0
        self.processMemoryGB = json["process_memory_gb"]?.doubleValue ?? 0

        self.gpuMemoryUsedMB = json["gpu_memory_used_mb"]?.intValue ?? 0
        self.gpuMemoryReservedOrDriverMB =
            json["gpu_memory_reserved_mb"]?.intValue
            ?? json["gpu_memory_driver_mb"]?.intValue
            ?? 0
    }

    /// Human friendly label for the device row in the status bar.
    var deviceLabel: String {
        switch gpuBackend {
        case "mps": return "Apple Silicon (MPS)"
        case "cuda": return "CUDA (\(device))"
        default: return "CPU"
        }
    }
}
