#!/usr/bin/env python3
"""Push the normalized mail corpus into Onyx via its public ingestion API.

Onyx's EML parser drops From/Subject/Date and never sets a link, so we do not
upload .eml files: each message becomes one structured document with a stable
id, a permalink section link and the mail headers as metadata. Resending an id
updates the document, so this is safe to rerun.

    python3 scripts/ingest_onyx.py --health-check
    python3 scripts/ingest_onyx.py --limit 50
    python3 scripts/ingest_onyx.py            # full corpus

Env: ONYX_URL, ONYX_API_KEY, DATA_DIR, MAIL_PUBLIC_URL, ONYX_CC_PAIR_ID (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ONYX_URL = os.environ.get("ONYX_URL", "http://localhost:8080").rstrip("/")
ONYX_KEY = os.environ.get("ONYX_API_KEY", "")
DATA = Path(os.environ.get("DATA_DIR", "data"))
MAIL_PUBLIC = os.environ.get("MAIL_PUBLIC_URL", "http://localhost:8081").rstrip("/")
CC_PAIR = os.environ.get("ONYX_CC_PAIR_ID", "").strip()


def headers() -> dict:
    if not ONYX_KEY:
        sys.exit("ONYX_API_KEY is not set (Onyx admin/PAT key with manage:connectors)")
    return {"Authorization": f"Bearer {ONYX_KEY}", "Content-Type": "application/json"}


def wait_for_health(timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{ONYX_URL}/health", timeout=5)
            if r.ok:
                print(f"onyx ready: {r.json()}")
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    sys.exit(f"onyx not healthy after {timeout}s at {ONYX_URL}/health")


def document(rec: dict) -> dict:
    doc = {
        "id": rec["id"],
        "source": "file",
        "semantic_identifier": rec["subject"][:250] or rec["id"],
        "title": rec["subject"][:250],
        "metadata": {
            "message_id": rec["message_id"],
            "thread_id": rec["thread_id"],
            "date": rec["date"],
            "from": rec["from"]["email"],
            "to": ", ".join(p["email"] for p in rec["to"]),
            "lang": rec["lang"],
            "source": rec["source"],
            "permalink": rec["permalink"],
            "synthetic_date": str(rec["synthetic_date"]).lower(),
        },
        "sections": [
            {
                "type": "text",
                "text": f'{rec["subject"]}\n\n{rec["body"]}',
                "link": f'{MAIL_PUBLIC}{rec["permalink"]}',
            }
        ],
        "doc_created_at": rec["date"],
        "doc_updated_at": rec["date"],
    }
    return doc


def push(rec: dict, retries: int = 4) -> str:
    payload: dict = {"document": document(rec)}
    if CC_PAIR:
        payload["cc_pair_id"] = int(CC_PAIR)
    last = ""
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{ONYX_URL}/onyx-api/ingestion", headers=headers(), json=payload, timeout=300
            )
            if r.status_code in (200, 201):
                return "ok"
            if r.status_code in (401, 403):
                sys.exit(f"auth rejected ({r.status_code}): {r.text[:300]}")
            last = f"{r.status_code} {r.text[:200]}"
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            return f"fail {last}"
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(2**attempt)
    return f"fail {last}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--health-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.health_check:
        wait_for_health(120)
        return 0

    manifest = DATA / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"missing {manifest}; run scripts/fetch-data.sh first")

    records = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    records = records[args.offset :]
    if args.limit:
        records = records[: args.limit]
    print(f"ingesting {len(records)} documents into {ONYX_URL} (cc_pair={CC_PAIR or 'default'})")

    if args.dry_run:
        print(json.dumps(document(records[0]), ensure_ascii=False, indent=2)[:1500])
        return 0

    ok = failed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(push, rec): rec for rec in records}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result == "ok":
                ok += 1
            else:
                failed += 1
                print(f"[{i}/{len(records)}] {futures[fut]['id']}: {result}", file=sys.stderr)
            if i % 100 == 0 or i == len(records):
                rate = i / max(time.time() - started, 0.1)
                print(f"  {i}/{len(records)} ok={ok} fail={failed} ({rate:.1f} doc/s)")

    print(f"done: ok={ok} failed={failed} in {time.time() - started:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
