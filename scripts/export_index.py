#!/usr/bin/env python3
"""Export the Onyx OpenSearch index to numpy vectors + JSONL metadata.

    python3 scripts/export_index.py --out /path/vectors

Produces `meta.jsonl` (id, document_id, title, blurb, content) and `vectors.npy`
(float32, L2-normalised, row order == meta order). Run on the host that can
reach OpenSearch; the files are portable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OS = os.environ.get("OPENSEARCH_URL", "https://localhost:9200").rstrip("/")
USER = os.environ.get("OPENSEARCH_USER", "admin")
PASS = os.environ.get("OPENSEARCH_PASSWORD", "")
INDEX = os.environ.get("INDEX_NAME", "onyx_chunks")


def fetch_all() -> list[dict]:
    auth = (USER, PASS) if PASS else None
    url = f"{OS}/{INDEX}/_search?scroll=2m&size=500"
    body = {
        "_source": ["document_id", "title", "blurb", "content", "content_vector"],
        "query": {"match_all": {}},
    }
    r = requests.post(url, json=body, auth=auth, verify=False, timeout=120)
    r.raise_for_status()
    payload = r.json()
    scroll_id = payload["_scroll_id"]
    hits = payload["hits"]["hits"]
    out = list(hits)
    while hits:
        r = requests.post(
            f"{OS}/_search/scroll",
            json={"scroll": "2m", "scroll_id": scroll_id},
            auth=auth,
            verify=False,
            timeout=120,
        )
        r.raise_for_status()
        payload = r.json()
        scroll_id = payload.get("_scroll_id", scroll_id)
        hits = payload["hits"]["hits"]
        out.extend(hits)
        print(f"  fetched {len(out)}", file=sys.stderr)
    requests.delete(
        f"{OS}/_search/scroll",
        json={"scroll_id": scroll_id},
        auth=auth,
        verify=False,
        timeout=30,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hits = fetch_all()
    meta, vecs = [], []
    for hit in hits:
        src = hit.get("_source", {})
        vec = src.get("content_vector")
        if not vec:
            continue
        meta.append(
            {
                "id": hit["_id"],
                "document_id": src.get("document_id"),
                "title": src.get("title") or "",
                "blurb": src.get("blurb") or "",
                "content": (src.get("content") or "")[:4000],
            }
        )
        vecs.append(vec)

    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    np.save(args.out / "vectors.npy", arr)
    with (args.out / "meta.jsonl").open("w", encoding="utf-8") as fh:
        for rec in meta:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"exported {len(meta)} rows, dim {arr.shape[1]}, {arr.nbytes/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
