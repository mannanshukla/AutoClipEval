#!/usr/bin/env bash
# download_playlist.sh
# Usage: ./download_playlist.sh "https://youtube.com/playlist?list=XXXX"

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PLAYLIST_URL"
  exit 1
fi

PLAYLIST_URL="$1"

yt-dlp \
  --ignore-errors \
  --yes-playlist \
  --format "bestvideo[height<=360]+bestaudio/best[height<=360]" \
  --merge-output-format mp4 \
  --write-auto-sub \
  --sub-lang "en" \
  --convert-subs vtt \
  --restrict-filenames \
  --output "%(title)s/%(title)s.%(ext)s" \
  "$PLAYLIST_URL"

# Rename English subtitle files from *.en.vtt → *.en-orig.vtt
find . -type f -name "*.en.vtt" -print0 |
  while IFS= read -r -d '' sub; do
    mv "$sub" "${sub%.en.vtt}.en-orig.vtt"
  done
