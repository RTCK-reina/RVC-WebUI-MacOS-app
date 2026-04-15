import Foundation

/// Live progress for a single backend task — drives the on-screen progress bar.
struct TaskProgress: Identifiable, Equatable {
    let id: String
    var percent: Double
    var message: String
    var phase: String
    var startTime: Date
    var lastUpdate: Date

    /// Heuristic time-remaining estimate based on elapsed + percent.
    var estimatedRemaining: TimeInterval? {
        guard percent > 1, percent < 100 else { return nil }
        let elapsed = lastUpdate.timeIntervalSince(startTime)
        let total = elapsed / (percent / 100.0)
        return max(0, total - elapsed)
    }

    init(fromParams json: JSONValue) {
        let now = Date()
        self.id = json["task_id"]?.stringValue ?? UUID().uuidString
        self.percent = json["percent"]?.doubleValue ?? 0
        self.message = json["message"]?.stringValue ?? ""
        self.phase = json["phase"]?.stringValue ?? ""
        let tsVal = json["timestamp"]?.doubleValue ?? now.timeIntervalSince1970
        self.lastUpdate = Date(timeIntervalSince1970: tsVal)
        self.startTime = self.lastUpdate
    }

    /// Merge a newer progress update into this record, keeping the original
    /// startTime so ETA estimates remain meaningful.
    mutating func update(with json: JSONValue) {
        self.percent = json["percent"]?.doubleValue ?? self.percent
        self.message = json["message"]?.stringValue ?? self.message
        self.phase = json["phase"]?.stringValue ?? self.phase
        let tsVal = json["timestamp"]?.doubleValue ?? Date().timeIntervalSince1970
        self.lastUpdate = Date(timeIntervalSince1970: tsVal)
    }
}
