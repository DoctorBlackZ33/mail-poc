"""Self-contained retrieval: numpy vectors + BM25, CPU embeddings.

Used on the frontend host after the Onyx/OpenSearch box is gone. Loads the
exported `vectors.npy` + `meta.jsonl`, embeds the query through a local CPU
llama-server, fuses BM25 and cosine with RRF, and returns the same hit shape as
the main retrieval service.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from rank_bm25 import BM25Okapi

EMBED_URL = os.environ.get("EMBED_URL", "http://embed:8080/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen3-Embedding-0.6B")
VECTORS = Path(os.environ.get("VECTORS_DIR", "/vectors"))
DATA = Path(os.environ.get("DATA_DIR", "/data"))
TOPK = int(os.environ.get("CANDIDATES", "30"))
TOKEN = re.compile(r"[0-9a-zäöüß]+", re.I)

app = FastAPI(title="mailrag-retrieval-local", version="0.1.0")
meta: list[dict] = []
vectors: np.ndarray | None = None
bm25: BM25Okapi | None = None
messages: dict[str, dict] = {}


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@app.on_event("startup")
def load() -> None:
    global vectors, bm25
    vectors = np.load(VECTORS / "vectors.npy")
    with (VECTORS / "meta.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            meta.append(json.loads(line))
    corpus = [tokenize(m["title"] + " " + m["content"]) for m in meta]
    bm25 = BM25Okapi(corpus)
    with (DATA / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            messages[rec["id"]] = rec


def embed(text: str) -> np.ndarray:
    r = requests.post(
        EMBED_URL, json={"model": EMBED_MODEL, "input": [text]}, timeout=120
    )
    r.raise_for_status()
    vec = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "messages": len(messages),
        "chunks": len(meta),
        "dim": int(vectors.shape[1]) if vectors is not None else None,
        "embed": EMBED_URL,
    }


@app.get("/search")
def search(q: str, k: int = Query(6, le=50)) -> dict:
    try:
        qv = embed(q)
    except requests.RequestException as exc:
        raise HTTPException(502, f"embedding failed: {exc}") from exc

    assert vectors is not None and bm25 is not None
    cosine = vectors @ qv
    dense_rank = np.argsort(-cosine)[:TOPK].tolist()
    bm_scores = bm25.get_scores(tokenize(q))
    sparse_rank = np.argsort(-bm_scores)[:TOPK].tolist()

    hits, seen = [], set()
    for idx, score in rrf([sparse_rank, dense_rank]):
        rec_meta = meta[idx]
        mid = unquote(str(rec_meta.get("document_id") or ""))
        rec = messages.get(mid)
        if not rec or mid in seen:
            continue
        seen.add(mid)
        hits.append(
            {
                "id": rec["id"],
                "score": round(float(score), 4),
                "thread_id": rec["thread_id"],
                "subject": rec["subject"],
                "date": rec["date"],
                "synthetic_date": rec["synthetic_date"],
                "lang": rec["lang"],
                "source": rec["source"],
                "from": rec["from"],
                "snippet": re.sub(r"\s+", " ", rec_meta["content"]).strip()[:260],
                "permalink": rec["permalink"],
            }
        )
        if len(hits) >= k:
            break
    return {"query": q, "mode": "local-bm25+dense", "hits": hits}
