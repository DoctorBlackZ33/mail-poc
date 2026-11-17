#!/usr/bin/env bash
# One-shot data fetch for the email-RAG POC.
#   ./scripts/fetch-data.sh                 # datasets -> ./data (EML + manifest)
#   ./scripts/fetch-data.sh --images        # also pull docker images
#   ./scripts/fetch-data.sh --models        # also pre-download GGUF weights (optional)
#
# Everything lands in ${DATA_DIR:-./data}; no network access is needed afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
MAILEX_REPO="${MAILEX_REPO:-https://github.com/youyou354/NLP_Mail_extraction.git}"
MAILEX_REV="${MAILEX_REV:-main}"
OMQ_URL="${OMQ_URL:-https://gueneumann.github.io/resources/omq_public_email_data.zip}"

PULL_IMAGES=0
PULL_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --images) PULL_IMAGES=1 ;;
    --models) PULL_MODELS=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

need() { command -v "$1" >/dev/null || { echo "missing dependency: $1" >&2; exit 1; }; }
need git; need curl; need python3; need unzip

mkdir -p "$DATA_DIR/src"
echo "==> data dir: $DATA_DIR"

# --- MailEx (Enron-derived, 1.5k threads / ~4k messages, thread bundles) -----
if [ ! -d "$DATA_DIR/src/mailex/.git" ]; then
  echo "==> cloning MailEx ($MAILEX_REV)"
  git clone --depth 1 --branch "$MAILEX_REV" "$MAILEX_REPO" "$DATA_DIR/src/mailex"
else
  echo "==> MailEx already present, skipping clone"
fi
MAILEX_SHA="$(git -C "$DATA_DIR/src/mailex" rev-parse HEAD)"

# --- OMQ (627 German support mails, CC BY-NC-SA 4.0) ------------------------
if [ ! -s "$DATA_DIR/src/omq.zip" ]; then
  echo "==> downloading OMQ"
  curl -fL --retry 3 -o "$DATA_DIR/src/omq.zip.part" "$OMQ_URL"
  mv "$DATA_DIR/src/omq.zip.part" "$DATA_DIR/src/omq.zip"
fi
mkdir -p "$DATA_DIR/src/omq"
unzip -o -q "$DATA_DIR/src/omq.zip" -d "$DATA_DIR/src/omq"

# --- normalize to EML + manifest -------------------------------------------
echo "==> building EML + manifest.jsonl"
python3 "$ROOT/scripts/prepare_mail.py" \
  --mailex "$DATA_DIR/src/mailex/data/raw_threads" \
  --mailex-full "$DATA_DIR/src/mailex/data/full_data" \
  --omq "$DATA_DIR/src/omq/omq_public_interactions.xml" \
  --out "$DATA_DIR"

cat > "$DATA_DIR/SOURCES.txt" <<EOF
fetched_at  $(date -u +%Y-%m-%dT%H:%M:%SZ)
mailex_repo $MAILEX_REPO
mailex_rev  $MAILEX_REV
mailex_sha  $MAILEX_SHA
omq_url     $OMQ_URL
omq_license CC-BY-NC-SA-4.0 (research/non-commercial)
mailex_note Enron-derived; Enron has no standard reuse license and contains real PII.
            Keep this corpus local. Do not redistribute. MailEx ships no dates;
            prepare_mail.py synthesizes them and marks X-Date-Synthetic.
EOF

if [ "$PULL_IMAGES" = 1 ]; then
  echo "==> pulling docker images"
  (cd "$ROOT/deploy/backend" && docker compose pull --ignore-buildable) || true
  (cd "$ROOT/deploy/frontend" && docker compose pull --ignore-buildable) || true
fi

if [ "$PULL_MODELS" = 1 ]; then
  echo "==> optional GGUF weights (only if you serve inference with llama.cpp)"
  need hf
  mkdir -p "$DATA_DIR/models"
  hf download ggml-org/gemma-4-E4B-it-GGUF --include '*Q4_K_M*' --local-dir "$DATA_DIR/models/gemma-4-e4b" || \
    echo "warn: gemma-4 E4B GGUF not fetched; check the repo id" >&2
  hf download Qwen/Qwen3-Reranker-0.6B-GGUF --include '*Q8_0*' --local-dir "$DATA_DIR/models/qwen3-reranker-0.6b" || \
    echo "warn: Qwen3-Reranker GGUF not fetched; check the repo id" >&2
fi

echo
echo "==> done"
wc -l < "$DATA_DIR/manifest.jsonl" | xargs -I{} echo "    messages: {}"
ls "$DATA_DIR/mail" | wc -l | xargs -I{} echo "    eml files: {}"
du -sh "$DATA_DIR" | awk '{print "    size: "$1}'
echo "    next: docker compose -f deploy/frontend/compose.yaml up -d --build"
