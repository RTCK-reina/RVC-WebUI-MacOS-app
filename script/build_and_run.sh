#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT_DIR/RVCApp/RVCApp.xcodeproj"
SCHEME="RVCApp"
CONFIGURATION="Debug"
DERIVED_DATA="$ROOT_DIR/build/DerivedData"
APP_NAME="RVC Swift"
APP_BUNDLE="$ROOT_DIR/build/$CONFIGURATION/$APP_NAME.app"
BUNDLE_ID="app.rvc.webui"

stop_app() {
  /usr/bin/osascript -e "tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true
  sleep 0.5
  /usr/bin/pkill -x "$APP_NAME" >/dev/null 2>&1 || true
}

build_app() {
  /usr/bin/xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -derivedDataPath "$DERIVED_DATA" \
    build
}

open_app() {
  /usr/bin/open -n "$APP_BUNDLE"
}

verify_app() {
  sleep 3
  /usr/bin/pgrep -x "$APP_NAME" >/dev/null
}

case "$MODE" in
  run|--run)
    stop_app
    build_app
    open_app
    ;;
  verify|--verify)
    stop_app
    build_app
    open_app
    verify_app
    ;;
  debug|--debug)
    stop_app
    build_app
    /usr/bin/lldb -- "$APP_BUNDLE/Contents/MacOS/$APP_NAME"
    ;;
  logs|--logs)
    stop_app
    build_app
    open_app
    /usr/bin/log stream --info --style compact --predicate "process == \"$APP_NAME\""
    ;;
  telemetry|--telemetry)
    stop_app
    build_app
    open_app
    /usr/bin/log stream --info --style compact --predicate "subsystem == \"$BUNDLE_ID\""
    ;;
  *)
    echo "usage: $0 [run|verify|debug|logs|telemetry]" >&2
    exit 2
    ;;
esac
