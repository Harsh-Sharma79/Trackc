#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ffmpeg -y \
  -i "$ROOT/public/how-to-use-silent.mp4" \
  -i "$ROOT/public/how-to-use-narration.wav" \
  -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=7.5,format=yuv420p[v]" \
  -map "[v]" -map 1:a:0 \
  -c:v libx264 -preset medium -crf 20 \
  -c:a aac -b:a 128k -shortest -movflags +faststart \
  "$ROOT/public/how-to-use.mp4"

echo "Wrote $ROOT/public/how-to-use.mp4"
