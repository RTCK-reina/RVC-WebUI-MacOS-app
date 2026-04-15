import SwiftUI
import AVFoundation

/// Inline audio preview using AVAudioPlayer — sufficient for WAV/FLAC/MP3/M4A.
struct AudioPlayerView: View {
    let url: URL
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false
    @State private var position: TimeInterval = 0
    @State private var duration: TimeInterval = 0
    @State private var timer: Timer?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button(action: toggle) {
                    Image(systemName: isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 28))
                }
                .buttonStyle(.borderless)

                Text((url.path as NSString).lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Text(timeString(position) + " / " + timeString(duration))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { position },
                    set: { seek($0) }),
                in: 0...max(duration, 0.1))
        }
        .padding(10)
        .background(.thinMaterial, in: .rect(cornerRadius: 8))
        .onAppear(perform: load)
        .onDisappear(perform: stop)
        .onChange(of: url) { _ in load() }
    }

    private func load() {
        stop()
        do {
            let p = try AVAudioPlayer(contentsOf: url)
            p.prepareToPlay()
            self.player = p
            self.duration = p.duration
            self.position = 0
        } catch {
            self.player = nil
        }
    }

    private func toggle() {
        guard let p = player else { return }
        if p.isPlaying {
            p.pause()
            isPlaying = false
            timer?.invalidate()
        } else {
            p.play()
            isPlaying = true
            startTimer()
        }
    }

    private func seek(_ t: TimeInterval) {
        guard let p = player else { return }
        p.currentTime = t
        position = t
    }

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            Task { @MainActor in
                guard let p = self.player else { return }
                self.position = p.currentTime
                if !p.isPlaying {
                    self.isPlaying = false
                    self.timer?.invalidate()
                }
            }
        }
    }

    private func stop() {
        player?.stop()
        timer?.invalidate()
        timer = nil
        isPlaying = false
    }

    private func timeString(_ t: TimeInterval) -> String {
        let s = Int(t)
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
