#!/usr/bin/env bash
# One-shot installer: corpus -> EML/manifest -> vectors -> stack.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

die() { echo "error: $*" >&2; exit 1; }
command -v docker >/dev/null || die "docker is required"
docker info >/dev/null 2>&1 || die "docker daemon not reachable"
docker compose version >/dev/null || die "docker compose plugin is required"
command -v curl >/dev/null || die "curl is required"
command -v python3 >/dev/null || die "python3 is required (used by the data fetcher)"

mkdir -p models vectors
MODEL=models/Qwen3-Embedding-0.6B-Q8_0.gguf
if [ ! -s "$MODEL" ]; then
  echo "==> downloading embedder (~610 MB)"
  curl -fL --retry 3 -o "$MODEL.part" \
    "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"
  mv "$MODEL.part" "$MODEL"
fi

[ -f data/manifest.jsonl ] || ./scripts/fetch-data.sh

echo "==> starting mail + embedding services"
docker compose up -d mail embed

echo "==> building vectors (first run embeds every message; CPU, minutes)"
docker compose --profile setup run --rm vectors

echo "==> starting retrieval + ui"
docker compose up -d retrieval ui

echo
echo "UI:        http://localhost:${UI_PORT:-9001}"
echo "Permalink: http://localhost:${MAIL_PORT:-9000}/mail/<message_id>"
echo "Optional LLM: set LLM_BASE_URL/LLM_MODEL in .env and restart ui"
echo "Optional auth proxy: cp Caddyfile.example Caddyfile && docker compose --profile proxy up -d"
