-- mail-simple schema. Requires Postgres 16 + pgvector (one extension).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id          text PRIMARY KEY,              -- Message-ID (or a hash for synthetic ids)
    thread_id   text NOT NULL,                 -- root Message-ID of the thread
    message_id  text,
    in_reply_to text,
    refs        jsonb NOT NULL DEFAULT '[]',   -- References header, as a JSON array
    sent_at     timestamptz,
    from_addr   text,
    to_addrs    text,
    subject     text NOT NULL DEFAULT '',
    body        text NOT NULL DEFAULT '',
    embedding   vector(1024),                  -- Qwen3-Embedding-0.6B
    tsv         tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(subject, '') || ' ' || coalesce(body, ''))
                ) STORED
);

-- Dense index and lexical index. Both are plain Postgres objects.
CREATE INDEX IF NOT EXISTS messages_embedding_idx
    ON messages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS messages_tsv_idx
    ON messages USING gin (tsv);
CREATE INDEX IF NOT EXISTS messages_thread_idx
    ON messages (thread_id, sent_at);
