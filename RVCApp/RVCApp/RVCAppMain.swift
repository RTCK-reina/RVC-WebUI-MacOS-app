import SwiftUI

@main
struct RVCApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var bridge = PythonBridge()
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup("RVC") {
            ContentView()
                .environmentObject(bridge)
                .environmentObject(appState)
                .task {
                    AppNotification.requestPermission()
                    delegate.bridge = bridge
                    await appState.bootstrap(bridge: bridge)
                }
                .frame(minWidth: 1040, minHeight: 720)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var bridge: PythonBridge?

    /// Close the red button = quit the app (not just hide the window).
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    /// Show a confirmation alert when there are active tasks.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let bridge, !bridge.activeProgress.isEmpty else {
            return .terminateNow
        }
        let count = bridge.activeProgress.count
        Task { @MainActor in
            let alert = NSAlert()
            alert.messageText = "処理中のタスクがあります"
            alert.informativeText = "\(count)件のタスクが実行中です。終了するとすべて中断されます。"
            alert.alertStyle = .warning
            alert.addButton(withTitle: "終了する")
            alert.addButton(withTitle: "キャンセル")
            let response = alert.runModal()
            if response == .alertFirstButtonReturn {
                sender.reply(toApplicationShouldTerminate: true)
            } else {
                sender.reply(toApplicationShouldTerminate: false)
            }
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        bridge?.killSync()
    }
}
