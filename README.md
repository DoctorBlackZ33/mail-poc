# mail-poc

Ask questions over a mailbox, get answers with clickable `date → mail` citations,
and draft replies. Split-pane Streamlit UI, self-hosted, CPU-only embeddings.

## Quick start

```bash
git clone https://github.com/DoctorBlackZ33/mail-poc.git
cd mail-poc
./install.sh
```

- UI: http://localhost:9001
- Permalinks: http://localhost:9000/mail/<message_id>

Optional chat model (any OpenAI-compatible endpoint):

```bash
cp .env.example .env
# set LLM_BASE_URL=http://host:8080/v1 and LLM_MODEL=<model-id>
docker compose up -d ui
```

Without an LLM the UI still retrieves mail and renders a lexical answer plus the
cited sources; drafts fall back to a skeleton.

## How it works

1. `scripts/fetch-data.sh` downloads **MailEx** (Enron-derived, threaded) and
   **OMQ** (German support mail) and normalizes every message to EML +
   `data/manifest.jsonl`.
2. `services/embed` runs **Qwen3-Embedding-0.6B** (Q8, CPU) via llama.cpp.
   `scripts/build_vectors.py` embeds the corpus into `vectors/` (18 MB for 4.5k mails).
3. `services/retrieval_local` fuses BM25 and cosine similarity (RRF) — no GPU,
   no external vector database.
4. `services/ui` shows chat on the left and a conventional mailbox on the right
   (date, language, subject, message count; search, EN/DE filter, mixed sort).
   Citations link to the mail permalink.

The default build yields **4,503 messages / 2,127 threads** (3,876 EN, 627 DE).

## Layout

```
install.sh                 corpus + vectors + stack
docker-compose.yml         mail, embed, retrieval, ui (+ setup/profile services)
services/mail/             corpus API, threads, permalinks, drafts
services/retrieval_local/  BM25 + numpy vectors (self-contained)
services/ui/               Streamlit UI
scripts/                   fetch, normalize, vector build, export
deploy/                    optional Onyx backend, llama.cpp compose, llama-swap example
```

Optional:

- `docker compose --profile proxy up -d` — Caddy on `:9080`, single origin for
  `/` and `/mail/*`, basic-auth block ready in `Caddyfile.example`.
- `deploy/onyx/` — Onyx + OpenSearch stack and the adapter in
  `services/retrieval_onyx/` if you want a heavier ingestion/search backend.
- `deploy/compose.rocm.yaml`, `deploy/compose.cuda.yaml`,
  `deploy/llama-swap.example.yaml` — GPU serving pattern where llama-swap only
  routes to dedicated per-model compose stacks.

## Data and licensing

| Source | What | Terms |
|---|---|---|
| [MailEx](https://github.com/youyou354/NLP_Mail_extraction) | 1.5k Enron threads | repo CC BY-SA 4.0; Enron-derived |
| [Enron](https://www.cs.cmu.edu/~enron/) | original corpus | research release, no standard license, real PII |
| [OMQ](https://gueneumann.github.io/resources/omqdata.html) | 627 German support mails | CC BY-NC-SA 4.0 |

No corpus data is committed. Keep `data/` local; MailEx/Enron contain real
personal data and must not be redistributed. MailEx ships no timestamps, so its
dates are generated deterministically and marked `*` in the UI.

## Stack

Streamlit · FastAPI · Caddy (optional) · llama.cpp · Qwen3-Embedding-0.6B ·
numpy + rank-bm25 · Docker Compose.
