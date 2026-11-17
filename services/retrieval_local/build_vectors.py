#!/usr/bin/env python3
"""Build vectors.npy + meta.jsonl from data/manifest.jsonl using the local
embedding server. Run after `docker compose up -d mail embed`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

EMBED_URL = os.environ.get("EMBED_URL", "http://embed:8080/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen3-Embedding-0.6B")
DATA = Path(os.environ.get("DATA_DIR", "/data"))
OUT = Path(os.environ.get("VECTORS_DIR", "/vectors"))


def wait_for_embed(timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(EMBED_URL.replace("/embeddings", "/models"), timeout=5)
            if r.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    sys.exit(f"embedding server not ready at {EMBED_URL}")


def embed(texts: list[str]) -> np.ndarray:
    r = requests.post(
        EMBED_URL, json={"model": EMBED_MODEL, "input": texts}, timeout=300
    )
    r.raise_for_status()
    vecs = np.asarray([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N mails (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = DATA / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"missing {manifest}; run scripts/fetch-data.sh first")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "vectors.npy").exists() and not args.force:
        print("vectors already exist (use --force to rebuild)")
        return 0

    recs = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    if args.limit:
        recs = recs[: args.limit]
        print(f"limit: embedding {len(recs)} mails")
    wait_for_embed()

    meta, chunks = [], []
    started = time.time()
    for i in range(0, len(recs), args.batch):
        batch = recs[i : i + args.batch]
        texts = [(r["subject"] + "\n\n" + r["body"])[:4000] for r in batch]
        chunks.append(embed(texts))
        for rec in batch:
            meta.append(
                {
                    "id": rec["id"],
                    "document_id": rec["id"],
                    "title": rec["subject"],
                    "blurb": rec["body"][:400],
                    "content": rec["body"][:4000],
                }
            )
        done = min(i + args.batch, len(recs))
        if done % 100 < args.batch:
            rate = done / max(time.time() - started, 0.1)
            eta = (len(recs) - done) / rate / 60 if rate else 0
            print(f"  {done}/{len(recs)} ({rate:.2f}/s, ~{eta:.0f} min left, flush)")

    arr = np.vstack(chunks)
    np.save(OUT / "vectors.npy", arr)
    with (OUT / "meta.jsonl").open("w", encoding="utf-8") as fh:
        for rec in meta:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {arr.shape[0]} vectors, dim {arr.shape[1]}, {arr.nbytes/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
