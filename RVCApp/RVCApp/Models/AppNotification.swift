import UserNotifications

/// Thin wrapper around UNUserNotificationCenter for task-completion banners.
enum AppNotification {
    private static var authorized = false

    /// Request notification permission (call once at app launch).
    static func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(
            options: [.alert, .sound]
        ) { granted, _ in
            authorized = granted
        }
    }

    /// Post a local notification. No-op if the app is frontmost (macOS
    /// suppresses banners for the active app by default, which is fine).
    static func send(title: String, body: String, id: String? = nil) {
        guard authorized else { return }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        let request = UNNotificationRequest(
            identifier: id ?? UUID().uuidString,
            content: content,
            trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }
}
