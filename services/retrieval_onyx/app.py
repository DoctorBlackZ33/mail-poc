"""Retrieval adapter for the mail corpus.

Primary path (OPENSEARCH_URL set): query the Onyx-built index directly —
BM25 + kNN on `content_vector`, reciprocal-rank fusion, then Qwen reranking.
This avoids Onyx's LLM query-expansion/selection pipeline, which needs a chat
model per search and is slow/fragile for a fixed corpus.

Fallback (no OPENSEARCH_URL): POST to Onyx `/search` and rerank its passages.

Response shape matches the dummy mail service so the UI can switch backends
with one env var.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import urllib3
from fastapi import FastAPI, HTTPException, Query

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ONYX_URL = os.environ.get("ONYX_URL", "http://api_server:8080").rstrip("/")
ONYX_KEY = os.environ.get("ONYX_API_KEY", "")
OS_URL = os.environ.get("OPENSEARCH_URL", "").rstrip("/")
OS_USER = os.environ.get("OPENSEARCH_USER", "admin")
OS_PASS = os.environ.get("OPENSEARCH_PASSWORD", "")
INDEX = os.environ.get("INDEX_NAME", "onyx_chunks")
EMBED_URL = os.environ.get("EMBED_URL", "http://host.docker.internal:9292/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "mail-qwen3-embedding-0.6b")
RERANK_URL = os.environ.get("RERANK_URL", "").rstrip("/")
if RERANK_URL in {"none", "off"}:
    RERANK_URL = ""
RERANK_MODEL = os.environ.get("RERANK_MODEL", "mail-qwen3-reranker-0.6b")
CANDIDATES = int(os.environ.get("CANDIDATES", "30"))
DATA = Path(os.environ.get("DATA_DIR", "/data"))

app = FastAPI(title="mailrag-retrieval", version="0.2.0")
messages: dict[str, dict] = {}

SOURCE_FIELDS = ["document_id", "title", "blurb", "content", "source_type", "link"]


@app.on_event("startup")
def load() -> None:
    manifest = DATA / "manifest.jsonl"
    if not manifest.exists():
        raise RuntimeError(f"missing {manifest}")
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            messages[rec["id"]] = rec

    # llama.cpp pays a one-off ROCm shader/load cost on the first rerank;
    # warm both models off the request path.
    def _warm() -> None:
        try:
            vec = embed("warmup")
            os_search({"size": 1, "_source": ["document_id"], "query": {"knn": {"content_vector": {"vector": vec, "k": 1}}}})
            rerank("warmup", ["warmup"])
        except Exception:
            pass

    threading.Thread(target=_warm, daemon=True).start()


def embed(text: str) -> list[float]:
    r = requests.post(
        EMBED_URL, json={"model": EMBED_MODEL, "input": [text]}, timeout=120
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def os_search(body: dict) -> list[dict]:
    r = requests.post(
        f"{OS_URL}/{INDEX}/_search",
        json=body,
        auth=(OS_USER, OS_PASS) if OS_PASS else None,
        verify=False,
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("hits", {}).get("hits", [])


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for hits in rank_lists:
        for rank, doc_id in enumerate(hits):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def onyx_search(q: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {ONYX_KEY}"} if ONYX_KEY else {}
    r = requests.post(
        f"{ONYX_URL}/search", headers=headers, json={"query": q}, timeout=180
    )
    r.raise_for_status()
    return r.json().get("results", [])


def rerank(query: str, docs: list[str]) -> list[tuple[int, float]]:
    base = [(i, 0.0) for i in range(len(docs))]
    if not RERANK_URL or not docs:
        return base
    try:
        r = requests.post(
            RERANK_URL,
            json={"model": RERANK_MODEL, "query": query, "documents": docs},
            # best-effort: a wedged reranker must not stall the answer
            timeout=6,
        )
        r.raise_for_status()
        rows = r.json().get("results") or r.json().get("data") or []
        scored = [
            (int(row["index"]), float(row.get("relevance_score", row.get("score", 0.0))))
            for row in rows
            if isinstance(row.get("index"), int)
        ]
        if scored:
            return sorted(scored, key=lambda x: x[1], reverse=True)
    except Exception:
        pass
    return base


def hit_from_record(rec: dict, score: float, content: str) -> dict:
    return {
        "id": rec["id"],
        "score": round(score, 4),
        "thread_id": rec["thread_id"],
        "subject": rec["subject"],
        "date": rec["date"],
        "synthetic_date": rec["synthetic_date"],
        "lang": rec["lang"],
        "source": rec["source"],
        "from": rec["from"],
        "snippet": re.sub(r"\s+", " ", content).strip()[:260],
        "permalink": rec["permalink"],
    }


def search_opensearch(q: str, k: int) -> dict:
    vec = embed(q)
    bm25_body = {
        "size": CANDIDATES,
        "_source": SOURCE_FIELDS,
        "query": {
            "multi_match": {"query": q, "fields": ["title^2", "content"]}
        },
    }
    knn_body = {
        "size": CANDIDATES,
        "_source": SOURCE_FIELDS,
        "query": {"knn": {"content_vector": {"vector": vec, "k": CANDIDATES}}},
    }
    bm25 = os_search(bm25_body)
    knn = os_search(knn_body)
    by_id = {h["_id"]: h for h in bm25}
    by_id.update({h["_id"]: h for h in knn})
    fused = rrf(
        [[h["_id"] for h in bm25], [h["_id"] for h in knn]], k=60
    )

    candidates: list[tuple[dict, float]] = []
    seen_docs: set[str] = set()
    for doc_id, score in fused:
        src = by_id[doc_id].get("_source", {})
        mid = unquote(str(src.get("document_id") or ""))
        rec = messages.get(mid)
        if not rec or mid in seen_docs:
            continue
        seen_docs.add(mid)
        candidates.append((rec, score))

    # short snippets keep the reranker fast; the full body is only for the answer
    docs = [
        (rec["subject"] + "\n\n" + rec["body"])[:600] for rec, _ in candidates
    ]
    ranked = rerank(q, docs)[:k]

    hits = []
    for idx, rscore in ranked:
        rec, fuse = candidates[idx]
        hits.append(hit_from_record(rec, rscore or fuse, rec["body"]))
    return {"query": q, "mode": "opensearch+rerank" if RERANK_URL else "opensearch", "hits": hits}


def search_onyx(q: str, k: int) -> dict:
    results = onyx_search(q)[:CANDIDATES]
    docs = [res.get("content") or "" for res in results]
    ranked = rerank(q, docs)[:k]
    hits = []
    for idx, score in ranked:
        res = results[idx]
        mid = None
        link = res.get("link") or ""
        if link:
            tail = urlparse(link).path.rstrip("/").rsplit("/", 1)[-1]
            mid = unquote(tail) if tail in messages else None
        rec = messages.get(mid) if mid else None
        if rec:
            hits.append(hit_from_record(rec, score, res.get("content") or ""))
    return {"query": q, "mode": "onyx+rerank", "hits": hits}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "messages": len(messages),
        "backend": "opensearch" if OS_URL else "onyx",
        "index": INDEX if OS_URL else None,
        "rerank": bool(RERANK_URL),
    }


@app.get("/search")
def search(q: str, k: int = Query(6, le=50)) -> dict:
    try:
        return search_opensearch(q, k) if OS_URL else search_onyx(q, k)
    except requests.RequestException as exc:
        raise HTTPException(502, f"search failed: {exc}") from exc
