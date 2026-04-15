#!/bin/bash
# RVC-WebUI Docker ビルド＆起動スクリプト
cd "$(dirname "$0")"
mkdir -p weights opt

echo "🔨 Docker ビルド開始（初回はモデルDL含めて10-15分かかります）..."
/usr/local/bin/docker compose -f docker-compose.apple.yml up --build 2>&1 | tee /tmp/rvc-docker-build.log
