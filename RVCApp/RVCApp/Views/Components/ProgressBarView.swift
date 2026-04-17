import SwiftUI

/// Reusable progress card showing percent + message + elapsed/remaining time.
/// Use with a PythonBridge task ID.
struct ProgressBarView: View {
    @EnvironmentObject var bridge: PythonBridge
    let taskID: String
    var title: String = "処理中"
    var onCancel: (() -> Void)?

    var body: some View {
        if let progress = bridge.activeProgress[taskID] {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(title).font(.subheadline).bold()
                    Spacer()
                    Text("\(Int(progress.percent))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    if let onCancel {
                        Button(action: onCancel) {
                            Image(systemName: "xmark.circle.fill")
                        }
                        .buttonStyle(.borderless)
                        .help("キャンセル")
                    }
                }
                ProgressView(value: progress.percent, total: 100)
                    .progressViewStyle(.linear)
                HStack {
                    Text(progress.message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer()
                    etaLabel(progress)
                }
            }
            .padding(12)
            .background(.thinMaterial, in: .rect(cornerRadius: 8))
        }
    }

    @ViewBuilder
    private func etaLabel(_ p: TaskProgress) -> some View {
        if let remaining = p.estimatedRemaining, remaining > 1 {
            Text("残り " + formatSeconds(remaining))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private func formatSeconds(_ s: TimeInterval) -> String {
        let total = Int(s)
        let m = total / 60
        let sec = total % 60
        if m > 0 {
            return String(format: "%d:%02d", m, sec)
        }
        return "\(sec)s"
    }
}

/// List of all currently running tasks — useful on the dashboard or when
/// multiple operations overlap.
struct ActiveTasksList: View {
    @EnvironmentObject var bridge: PythonBridge

    var body: some View {
        VStack(spacing: 8) {
            ForEach(Array(bridge.activeProgress.values), id: \.id) { p in
                ProgressBarView(
                    taskID: p.id,
                    title: label(for: p.phase),
                    onCancel: { [taskID = p.id] in
                        Task {
                            _ = try? await bridge.callRaw(
                                "cancel",
                                params: .object(["task_id": .string(taskID)]))
                            bridge.clearProgress(for: taskID)
                        }
                    })
            }
        }
    }

    private func label(for phase: String) -> String {
        switch phase {
        case "inference": return "推論中"
        case "batch":     return "バッチ推論中"
        case "separation": return "音声分離中"
        case "training":  return "学習中"
        default:          return "処理中"
        }
    }
}
