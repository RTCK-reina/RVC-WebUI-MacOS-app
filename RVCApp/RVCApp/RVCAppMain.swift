import SwiftUI

@main
struct RVCApp: App {
    @StateObject private var bridge = PythonBridge()
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup("RVC") {
            ContentView()
                .environmentObject(bridge)
                .environmentObject(appState)
                .task {
                    await appState.bootstrap(bridge: bridge)
                }
                .frame(minWidth: 1040, minHeight: 720)
        }
        .windowStyle(.hiddenTitleBar)
    }
}
