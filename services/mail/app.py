"""Dummy mail service: serves the MailEx + OMQ corpus as threads, messages,
permalinks and drafts. Also provides a dependency-free lexical retriever so the
Streamlit UI runs standalone without Onyx.

Endpoints
  GET  /health
  GET  /api/threads?q=&limit=&offset=
  GET  /api/thread/{thread_id}
  GET  /api/mail/{message_id}
  GET  /mail/{message_id}              HTML permalink (the citation target)
  GET  /api/search?q=&k=               lexical fallback retrieval
  POST /api/draft                      {"thread_id","subject","body","in_reply_to"}
  GET  /api/drafts?thread_id=
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

DATA = Path(os.environ.get("DATA_DIR", "/data"))
MANIFEST = DATA / "manifest.jsonl"
DRAFTS = DATA / "drafts.jsonl"

TOKEN = re.compile(r"[0-9a-zäöüß]+", re.I)

app = FastAPI(title="dummy-mail", version="0.1.0")

messages: dict[str, dict] = {}
threads: dict[str, list[str]] = defaultdict(list)
postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
doc_len: dict[str, int] = {}
avg_len = 0.0


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@app.on_event("startup")
def load() -> None:
    global avg_len
    if not MANIFEST.exists():
        raise RuntimeError(f"missing {MANIFEST}; run scripts/fetch-data.sh")
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            messages[rec["id"]] = rec
            threads[rec["thread_id"]].append(rec["id"])
    for ids in threads.values():
        ids.sort(key=lambda i: messages[i]["date"])

    total = 0
    for mid, rec in messages.items():
        toks = tokenize(f'{rec["subject"]} {rec["body"]}')
        doc_len[mid] = len(toks) or 1
        total += doc_len[mid]
        for term, tf in Counter(toks).items():
            postings[term].append((mid, tf))
    avg_len = total / max(len(messages), 1)

    for ids in threads.values():
        ids.sort(key=lambda i: messages[i]["date"])


def thread_summary(tid: str) -> dict:
    ids = threads[tid]
    head = messages[ids[-1]]
    people = {f'{m["from"]["name"]} <{m["from"]["email"]}>' for m in (messages[i] for i in ids)}
    return {
        "thread_id": tid,
        "subject": head["subject"],
        "messages": len(ids),
        "last_date": head["date"],
        "first_date": messages[ids[0]]["date"],
        "participants": sorted(people)[:6],
        "lang": head["lang"],
        "source": head["source"],
        "synthetic_date": any(messages[i]["synthetic_date"] for i in ids),
    }


def lexical_search(q: str, k: int) -> list[dict]:
    terms = tokenize(q)
    if not terms:
        return []
    scores: dict[str, float] = defaultdict(float)
    n = len(messages)
    for term in set(terms):
        plist = postings.get(term, [])
        idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
        for mid, tf in plist:
            norm = tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * doc_len[mid] / avg_len))
            scores[mid] += idf * norm
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for mid, score in ranked:
        rec = messages[mid]
        out.append(
            {
                "id": mid,
                "score": round(score, 4),
                "thread_id": rec["thread_id"],
                "subject": rec["subject"],
                "date": rec["date"],
                "synthetic_date": rec["synthetic_date"],
                "lang": rec["lang"],
                "source": rec["source"],
                "from": rec["from"],
                "snippet": make_snippet(rec["body"], terms),
                "permalink": rec["permalink"],
            }
        )
    return out


def make_snippet(body: str, terms: list[str], width: int = 260) -> str:
    flat = re.sub(r"\s+", " ", body).strip()
    low = flat.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=-1)
    if pos < 0:
        return flat[:width] + ("…" if len(flat) > width else "")
    start = max(0, pos - width // 3)
    return ("…" if start else "") + flat[start : start + width] + ("…" if start + width < len(flat) else "")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "messages": len(messages), "threads": len(threads)}


@app.get("/api/threads")
def list_threads(
    q: str = "",
    limit: int = Query(50, le=500),
    offset: int = 0,
    lang: str | None = None,
    order: str = "desc",
    mode: str = "date",
) -> dict:
    if q.strip():
        hits = {h["thread_id"] for h in lexical_search(q, 400)}
        items = [thread_summary(t) for t in hits]
    else:
        items = [thread_summary(t) for t in threads]

    if lang and lang.lower() in {"en", "de"}:
        items = [t for t in items if t["lang"] == lang.lower()]

    reverse = order != "asc"
    items.sort(key=lambda t: t["last_date"], reverse=reverse)

    # Interleave one EN and one DE thread at a time, each still in date order,
    # so page 1 shows both languages instead of the newest language only.
    if mode == "mixed" and not lang:
        en = [t for t in items if t["lang"] == "en"]
        de = [t for t in items if t["lang"] == "de"]
        items = [x for pair in zip_longest(en, de) for x in pair if x]

    return {"total": len(items), "items": items[offset : offset + limit]}


@app.get("/api/thread/{thread_id}")
def get_thread(thread_id: str) -> dict:
    if thread_id not in threads:
        raise HTTPException(404, "unknown thread")
    return {
        "thread": thread_summary(thread_id),
        "messages": [messages[i] for i in threads[thread_id]],
    }


@app.get("/api/mail/{message_id}")
def get_mail(message_id: str) -> dict:
    if message_id not in messages:
        raise HTTPException(404, "unknown message")
    return messages[message_id]


@app.get("/api/search")
def search(q: str, k: int = Query(8, le=50)) -> dict:
    return {"query": q, "mode": "lexical", "hits": lexical_search(q, k)}


class Draft(BaseModel):
    thread_id: str
    subject: str
    body: str
    in_reply_to: str | None = None


@app.post("/api/draft")
def save_draft(draft: Draft) -> dict:
    if draft.thread_id not in threads:
        raise HTTPException(404, "unknown thread")
    rec = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **draft.model_dump(),
    }
    DRAFTS.parent.mkdir(parents=True, exist_ok=True)
    with DRAFTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "draft": rec}


@app.get("/api/drafts")
def list_drafts(thread_id: str | None = None) -> dict:
    if not DRAFTS.exists():
        return {"items": []}
    items = [json.loads(line) for line in DRAFTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if thread_id:
        items = [d for d in items if d["thread_id"] == thread_id]
    return {"items": items[-50:]}


@app.get("/mail/{message_id}", response_class=HTMLResponse)
def permalink(message_id: str) -> str:
    if message_id not in messages:
        raise HTTPException(404, "unknown message")
    m = messages[message_id]
    synthetic = (
        '<p class="warn">Date is synthetic (MailEx ships no timestamps).</p>'
        if m["synthetic_date"]
        else ""
    )
    body = html.escape(m["body"])
    people = ", ".join(
        html.escape(f'{p["name"]} <{p["email"]}>'.strip()) for p in [m["from"], *m["to"]]
    )
    return f"""<!doctype html>
<html lang="{m['lang']}"><head><meta charset="utf-8">
<title>{html.escape(m['subject'])}</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;max-width:820px;margin:40px auto;padding:0 16px;color:#1b1b1b}}
 header{{border-bottom:1px solid #ddd;margin-bottom:18px;padding-bottom:12px}}
 dl{{display:grid;grid-template-columns:90px 1fr;gap:4px 12px;font-size:14px}}
 dt{{color:#666}} dd{{margin:0}}
 pre{{white-space:pre-wrap;font:inherit}}
 .warn{{background:#fff3cd;border-left:4px solid #e0a800;padding:8px 12px;font-size:13px}}
 .tag{{display:inline-block;background:#eef;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:6px}}
</style></head><body>
<header><h1>{html.escape(m['subject'])}</h1>
<span class="tag">{m['source']}</span><span class="tag">{m['lang']}</span>
<span class="tag">thread {html.escape(m['thread_id'])}</span></header>
{synthetic}
<dl>
 <dt>Date</dt><dd>{html.escape(m['date'])}</dd>
 <dt>From / To</dt><dd>{people}</dd>
 <dt>Message-ID</dt><dd>{html.escape(m['message_id'])}</dd>
 <dt>In-Reply-To</dt><dd>{html.escape(m['in_reply_to'] or '—')}</dd>
</dl>
<h2>Body</h2><pre>{body}</pre>
<p><a href="/api/thread/{html.escape(m['thread_id'])}">thread JSON</a></p>
</body></html>"""


@app.exception_handler(404)
def not_found(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=404)
