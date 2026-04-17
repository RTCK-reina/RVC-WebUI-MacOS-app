import SwiftUI

/// Permanent bottom bar: shows backend status, device, CPU/MEM/GPU gauges.
/// Updates every second from the Python `resource_stats` notification.
struct StatusBarView: View {
    @EnvironmentObject var bridge: PythonBridge

    var body: some View {
        HStack(spacing: 16) {
            StatusDot(status: bridge.backendStatus)
            Text(statusLabel(bridge.backendStatus))
                .font(.caption)
                .frame(minWidth: 64, alignment: .leading)

            Divider().frame(height: 16)

            Gauge(label: "CPU",
                  value: bridge.resourceStats.cpuPercent,
                  max: 100,
                  unit: "%")

            Gauge(label: "MEM",
                  value: bridge.resourceStats.memoryUsedGB,
                  max: bridge.resourceStats.memoryTotalGB,
                  unit: "GB",
                  format: "%.1f")

            gpuGauge

            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.bar)
    }

    @ViewBuilder
    private var gpuGauge: some View {
        let stats = bridge.resourceStats
        if stats.gpuBackend == "mps" || stats.gpuBackend == "cuda" {
            HStack(spacing: 4) {
                Text("GPU")
                    .font(.caption).bold()
                    .foregroundStyle(.secondary)
                ProgressView(value: min(stats.gpuPercent, 100), total: 100)
                    .progressViewStyle(.linear)
                    .frame(width: 80)
                Text("\(String(format: "%.0f", stats.gpuPercent))%")
                    .font(.caption.monospacedDigit())
                    .frame(minWidth: 36, alignment: .trailing)
                Text("\(stats.gpuMemoryUsedMB)MB")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 52, alignment: .trailing)
            }
        }
    }

    private func statusLabel(_ s: String) -> String {
        switch s {
        case "starting":   return "起動中"
        case "idle":       return "待機"
        case "inferring":  return "推論中"
        case "separating": return "分離中"
        case "training":   return "学習中"
        case "realtime":   return "リアルタイム"
        case "shutting_down": return "終了中"
        default:           return s
        }
    }
}

private struct Gauge: View {
    let label: String
    let value: Double
    let max: Double
    let unit: String
    var format: String = "%.0f"

    var body: some View {
        HStack(spacing: 4) {
            Text(label).font(.caption).bold().foregroundStyle(.secondary)
            ProgressView(value: min(value, max), total: max > 0 ? max : 1)
                .progressViewStyle(.linear)
                .frame(width: 80)
            Text("\(String(format: format, value)) \(unit)")
                .font(.caption.monospacedDigit())
                .frame(minWidth: 70, alignment: .trailing)
        }
    }
}

private struct StatusDot: View {
    let status: String
    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 8, height: 8)
            .shadow(color: color.opacity(0.6), radius: 2)
    }
    private var color: Color {
        switch status {
        case "idle":      return .green
        case "starting":  return .yellow
        case "inferring", "separating", "training", "realtime":
            return .blue
        case "shutting_down":
            return .orange
        default: return .gray
        }
    }
}
