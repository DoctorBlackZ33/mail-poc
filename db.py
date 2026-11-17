"""Postgres access through the `psql` CLI.

No Python database driver: this module shells out to psql with `-f -`, so the
only requirement is the standard PostgreSQL client. Rows come back as JSON via
`row_to_json`, which keeps parsing trivial.
"""

from __future__ import annotations

import json
import subprocess


def psql(sql: str) -> str:
    proc = subprocess.run(
        ["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-f", "-"],
        input=sql,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout


def query(sql: str) -> list[dict]:
    return [json.loads(line) for line in psql(sql).splitlines() if line.strip()]
