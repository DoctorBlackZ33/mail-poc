#!/usr/bin/env python3
"""Normalize MailEx (EN, threaded) + OMQ (DE, support mail) into EML + manifest.jsonl.

Output:
  <out>/mail/<message_id>.eml      one RFC 2822 message per file, stable names
  <out>/manifest.jsonl             one JSON record per message, sorted by date
  <out>/SOURCES.txt                provenance written by fetch-data.sh

MailEx carries no dates. Dates are synthesized deterministically inside the
Enron era and marked `synthetic_date: true` so the UI can label them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

DELIM = re.compile(r"^-{5,}\s*$", re.M)
HEADER = re.compile(r"^(FROM|TO|CC|BCC|SUBJECT|DATE)\s*:\s*(.*)$", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def sha1(*parts: str) -> str:
    h = hashlib.sha1("\x1f".join(parts).encode("utf-8", "replace"))
    return h.hexdigest()


def addr(value: str, fallback_domain: str) -> tuple[str, str]:
    """'Blair, Lynn' -> ('', 'blair.lynn@enron.com'); 'a@b.c' -> ('', 'a@b.c')."""
    value = (value or "").strip().strip('"')
    found = EMAIL.search(value)
    if found:
        return "", found.group(0).lower()
    name = value.replace(",", " ").strip()
    slug = re.sub(r"[^a-z0-9]+", ".", name.lower()).strip(".") or "unknown"
    return name, f"{slug}@{fallback_domain}"


def parse_mailex_thread(text: str) -> list[dict]:
    """Split a MailEx raw_threads file into messages, newest first."""
    out: list[dict] = []
    for block in DELIM.split(text):
        block = block.strip()
        if not block:
            continue
        headers: dict[str, str] = {}
        body: list[str] = []
        in_body = False
        for line in block.splitlines():
            if not in_body:
                m = HEADER.match(line.strip())
                if m:
                    headers[m.group(1).upper()] = m.group(2).strip()
                    continue
                if not line.strip():
                    in_body = True
                    continue
                in_body = True
            body.append(line)
        body_text = "\n".join(body).strip()
        if not body_text:
            continue
        out.append({"headers": headers, "body": body_text})
    return out


def detokenize(tokens: list) -> str:
    """MailEx full_data stores tokenized turns; rebuild readable text."""
    out = ""
    for tok in tokens:
        tok = str(tok)
        if tok in ",.;:!?%)]}'\"":
            out += tok
        elif tok in "([{":
            out += (" " if out else "") + tok
        else:
            out += (" " if out else "") + tok
    return out.strip()


def mailex_full_threads(full_dir: Path) -> tuple[list[dict], list[str]]:
    """1.5k threads / ~4k turns without headers or dates: synthesise both."""
    msgs: list[dict] = []
    warnings: list[str] = []
    for path in sorted(full_dir.glob("*.json")):
        thread_id = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"mailex-full: {path.name}: {exc}")
            continue
        turns = data.get("sentences") or []
        if not turns:
            warnings.append(f"mailex-full: no turns in {path.name}")
            continue
        bodies = [detokenize(t) if isinstance(t, list) else str(t) for t in turns]
        seed = int(sha1("mailex-full", thread_id)[:8], 16)
        base = datetime(1999, 1, 4, tzinfo=timezone.utc) + timedelta(days=seed % 1000)
        ids = [f"<{sha1(thread_id, str(i), b)[:24]}@mailex.local>" for i, b in enumerate(bodies)]
        subject = " ".join(bodies[0].split()[:8]) if bodies[0] else thread_id
        for i, body in enumerate(bodies):
            if not body.strip():
                continue
            msgs.append(
                {
                    "id": ids[i].strip("<>"),
                    "thread_id": thread_id,
                    "message_id": ids[i],
                    "in_reply_to": ids[i - 1] if i else None,
                    "references": ids[:i],
                    "date": (base + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                    "synthetic_date": True,
                    "synthetic_headers": True,
                    "from": {"name": "Enron employee", "email": "employee@enron.example"},
                    "to": [{"name": "Enron team", "email": "team@enron.example"}],
                    "cc": [],
                    "subject": f"{subject}…" if len(bodies[0].split()) > 8 else subject,
                    "body": body,
                    "lang": "en",
                    "source": "mailex",
                    "folder": thread_id.rsplit("_", 1)[0],
                }
            )
    return msgs, warnings


def mailex_messages(thread_dir: Path) -> tuple[list[dict], list[str]]:
    msgs: list[dict] = []
    warnings: list[str] = []
    for path in sorted(thread_dir.iterdir()):
        if not path.is_file():
            continue
        thread_id = path.name
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = parse_mailex_thread(text)
        if not blocks:
            warnings.append(f"mailex: no messages in {path}")
            continue
        # Deterministic base date in the Enron era; first block is the newest.
        seed = int(sha1("mailex", thread_id)[:8], 16)
        newest = datetime(1999, 1, 4, tzinfo=timezone.utc) + timedelta(
            days=seed % 1000, hours=seed % 24
        )
        ids = [
            f"<{sha1(thread_id, str(i), b.get('body', ''))[:24]}@mailex.local>"
            for i, b in enumerate(blocks)
        ]
        for i, block in enumerate(blocks):
            h = block["headers"]
            date = newest - timedelta(days=i, hours=(i * 7) % 24)
            from_name, from_email = addr(h.get("FROM", ""), "enron.com")
            to_name, to_email = addr(h.get("TO", ""), "enron.com")
            cc_name, cc_email = addr(h.get("CC", ""), "enron.com")
            msgs.append(
                {
                    "id": ids[i].strip("<>"),
                    "thread_id": thread_id,
                    "message_id": ids[i],
                    "in_reply_to": ids[i + 1] if i + 1 < len(ids) else None,
                    "references": ids[i + 1 :],
                    "date": date.isoformat().replace("+00:00", "Z"),
                    "synthetic_date": True,
                    "from": {"name": from_name, "email": from_email},
                    "to": [{"name": to_name, "email": to_email}],
                    "cc": [{"name": cc_name, "email": cc_email}] if cc_email else [],
                    "subject": h.get("SUBJECT", "") or f"(no subject) {thread_id}",
                    "body": block["body"],
                    "lang": "en",
                    "source": "mailex",
                    "folder": thread_id.rsplit("_", 1)[0],
                }
            )
    return msgs, warnings


def decode_omq(path: Path) -> str:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", "replace")
    start = text.find("<dataset")
    return text[start:] if start >= 0 else text


def omq_messages(path: Path) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    text = decode_omq(path)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        warnings.append(f"omq: XML parse failed ({exc}); falling back to regex")
        return omq_regex_fallback(text), warnings

    msgs: list[dict] = []
    for interaction in root.iter("interaction"):
        meta = interaction.find("metadata")
        if meta is None:
            continue
        omq_id = (meta.findtext("id") or "").strip()
        date_raw = (meta.findtext("date") or "").strip()
        scenario = (meta.findtext("businessScenario") or "").strip()
        body_el = interaction.find("text")
        body = "".join(body_el.itertext()).strip() if body_el is not None else ""
        body = re.sub(r"[ \t]+\n", "\n", body)
        if not body:
            warnings.append(f"omq: empty body for id={omq_id}")
            continue
        try:
            date = datetime.fromisoformat(date_raw).replace(tzinfo=timezone.utc)
        except ValueError:
            date = datetime(2012, 1, 1, tzinfo=timezone.utc)
            warnings.append(f"omq: bad date {date_raw!r} id={omq_id}")
        subject = f"Support-Anfrage {omq_id} – {scenario}" if scenario else f"Support-Anfrage {omq_id}"
        msgs.append(
            {
                "id": f"omq-{omq_id}",
                "thread_id": f"omq-{omq_id}",
                "message_id": f"<omq-{omq_id}@omq.local>",
                "in_reply_to": None,
                "references": [],
                "date": date.isoformat().replace("+00:00", "Z"),
                "synthetic_date": False,
                "from": {"name": "Kunde", "email": "kunde@warehouse.example"},
                "to": [{"name": "Support", "email": "support@warehouse.example"}],
                "cc": [],
                "subject": subject,
                "body": body,
                "lang": "de",
                "source": "omq",
                "folder": "support",
            }
        )
    return msgs, warnings


def omq_regex_fallback(text: str) -> list[dict]:
    msgs: list[dict] = []
    for chunk in re.split(r"<interaction>", text)[1:]:
        omq_id = (re.search(r"<id>\s*([^<]+)\s*</id>", chunk) or [None, ""])[1].strip()
        date_raw = (re.search(r"<date>\s*([^<]+)\s*</date>", chunk) or [None, ""])[1].strip()
        body = re.sub(r"<[^>]+>", " ", chunk)
        body = re.sub(r"\s+", " ", body).strip()
        if not omq_id or not body:
            continue
        try:
            date = datetime.fromisoformat(date_raw).replace(tzinfo=timezone.utc)
        except ValueError:
            date = datetime(2012, 1, 1, tzinfo=timezone.utc)
        msgs.append(
            {
                "id": f"omq-{omq_id}",
                "thread_id": f"omq-{omq_id}",
                "message_id": f"<omq-{omq_id}@omq.local>",
                "in_reply_to": None,
                "references": [],
                "date": date.isoformat().replace("+00:00", "Z"),
                "synthetic_date": False,
                "from": {"name": "Kunde", "email": "kunde@warehouse.example"},
                "to": [{"name": "Support", "email": "support@warehouse.example"}],
                "cc": [],
                "subject": f"Support-Anfrage {omq_id}",
                "body": body,
                "lang": "de",
                "source": "omq",
                "folder": "support",
            }
        )
    return msgs


def to_eml(rec: dict) -> EmailMessage:
    msg = EmailMessage()
    msg["Message-ID"] = rec["message_id"]
    if rec["in_reply_to"]:
        msg["In-Reply-To"] = rec["in_reply_to"]
    if rec["references"]:
        msg["References"] = " ".join(rec["references"])
    msg["Date"] = format_datetime(datetime.fromisoformat(rec["date"].replace("Z", "+00:00")))
    msg["From"] = f'{rec["from"]["name"]} <{rec["from"]["email"]}>'.strip()
    msg["To"] = ", ".join(f'{p["name"]} <{p["email"]}>'.strip() for p in rec["to"])
    if rec["cc"]:
        msg["Cc"] = ", ".join(f'{p["name"]} <{p["email"]}>'.strip() for p in rec["cc"])
    msg["Subject"] = rec["subject"]
    msg["X-Thread-Id"] = rec["thread_id"]
    msg["X-Source"] = rec["source"]
    msg["X-Language"] = rec["lang"]
    if rec["synthetic_date"]:
        msg["X-Date-Synthetic"] = "true"
    msg.set_content(rec["body"])
    return msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailex", type=Path, help="MailEx data/raw_threads directory")
    ap.add_argument("--mailex-full", type=Path, help="MailEx data/full_data directory (tokenized turns)")
    ap.add_argument("--omq", type=Path, help="OMQ omq_public_interactions.xml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records: list[dict] = []
    warnings: list[str] = []
    full: list[dict] = []
    raw: list[dict] = []
    if args.mailex_full:
        full, warn = mailex_full_threads(args.mailex_full)
        warnings += warn
    if args.mailex:
        raw, warn = mailex_messages(args.mailex)
        warnings += warn
    # raw_threads carries real From/To/Subject, so it wins per thread
    raw_threads = {m["thread_id"] for m in raw}
    records += [m for m in full if m["thread_id"] not in raw_threads] + raw
    if args.omq:
        msgs, warn = omq_messages(args.omq)
        warnings += warn
        records += msgs
    if not records:
        print("no messages parsed — check --mailex / --mailex-full / --omq", file=sys.stderr)
        return 2

    mail_dir = args.out / "mail"
    mail_dir.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda r: (r["date"], r["id"]))

    seen: set[str] = set()
    for rec in records:
        assert rec["id"] not in seen, f"duplicate id {rec['id']}"
        seen.add(rec["id"])
        rec.setdefault("synthetic_headers", False)
        (mail_dir / f"{rec['id']}.eml").write_bytes(bytes(to_eml(rec)))

    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            rec["eml"] = f"mail/{rec['id']}.eml"
            rec["permalink"] = f"/mail/{rec['id']}"
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    threads = {r["thread_id"] for r in records}
    langs: dict[str, int] = {}
    srcs: dict[str, int] = {}
    for r in records:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
        srcs[r["source"]] = srcs.get(r["source"], 0) + 1

    print(f"messages : {len(records)}")
    print(f"threads  : {len(threads)}")
    print(f"languages: {langs}")
    print(f"sources  : {srcs}")
    print(f"out      : {args.out}")
    for w in warnings[:20]:
        print(f"warn: {w}", file=sys.stderr)
    if len(warnings) > 20:
        print(f"warn: {len(warnings) - 20} more suppressed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
