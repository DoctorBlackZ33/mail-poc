# mail-simple

Mail is the interface: pull a mailbox into Postgres/pgvector, ask a question by
mail, get a cited answer back by mail.

## What it does

1. **ingest** — read Maildir or IMAP, parse with Python's `email` module, embed
   every message, upsert into Postgres.
2. **ask** — embed the question, retrieve with BM25 + cosine (reciprocal-rank
   fusion in SQL), rerank, answer with the chat model.
3. **serve** — poll the mailbox; each new message is a question; the answer is
   sent as a reply in the same thread and stored for the next question.

## Quick start (Maildir, no mail server)

```bash
cp .env.example .env        # set POSTGRES_PASSWORD
docker compose up -d db
cp your-mail/*.eml maildir/new/
docker compose run --rm bot ingest
docker compose run --rm bot ask "what did we tell the customer about the invoice?"
```

`ingest` consumes the queue: files in `maildir/new/` are stored and moved to
`maildir/cur/`. To ask the bot something, drop a new file into `maildir/new/`
and run `docker compose run --rm bot serve --once`; the reply appears in
`maildir/Sent/` with `In-Reply-To`/`References` set.

## Real mailbox

Set `MAIL_SOURCE=imap` and fill `IMAP_*` / `SMTP_*` in `.env`. Ingestion reads
`IMAP_MAILBOX`; `serve` watches `IMAP_QUESTION_MAILBOX` for UNSEEN mail, so point
that at a folder where a server-side rule delivers questions (or use a dedicated
bot mailbox). Ingestion uses `imaplib`, replies use `smtplib`; both are standard
library. Gmail needs an app password; Microsoft 365 needs SMTP AUTH enabled.

```bash
docker compose run --rm bot ingest --since 01-Jan-2024
docker compose --profile bot up -d        # answers new mail every POLL_SECONDS
```

## Inference

The code talks to three OpenAI-compatible endpoints. Run them yourself with:

```bash
# put the GGUF files in ./models first
docker compose --profile llama up -d
```

or point `EMBED_URL`, `RERANK_URL`, `CHAT_URL` at another machine. Nothing is
loaded on the laptop unless you start that profile.

## How retrieval works

`retrieve()` is one SQL statement: the 30 nearest vectors and the 30 best
full-text matches, fused with `1/(60+rank)`. The reranker then orders the top
candidates; the chat model only sees those mails. No vector client, no search
service, no ranking library.

## Dependencies

| Dependency | Why it is safe to keep |
|---|---|
| Python 3 standard library | No packages, no lockfile. `email`, `imaplib`, `smtplib`, `urllib`, `csv`, `subprocess`. |
| `psql` (Postgres client) | Stable CLI; the DB contract is SQL. |
| PostgreSQL 16 + pgvector | Postgres is a standard; pgvector is a single, widely deployed extension. HNSW index is part of the schema. |
| llama.cpp | Used **only** through its OpenAI-compatible HTTP API (`/v1/embeddings`, `/v1/rerank`, `/v1/chat/completions`). The binary can be swapped or upgraded without touching the code. |
| Debian stable base image | Long-lived, boring, no build toolchain. |

Deliberately absent: web framework, ORM, vector SDK, mail SDK, task queue,
frontend build, Streamlit, Caddy. The whole bot is two Python files.

## Limits

- Attachments are not indexed.
- HTML mail is reduced to its text part.
- One mailbox, one bot address; no per-user permissions.
- `tsv` uses the `simple` text search config, so German is not stemmed.

Those are deliberate: this branch is the smallest thing that closes the loop.
