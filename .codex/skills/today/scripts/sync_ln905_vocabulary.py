#!/usr/bin/env python3
"""Upsert the managed LN905 vocabulary block into Neath; never deletes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


VAULT = Path(__file__).resolve().parents[4]
BANK = VAULT / "99_学习情况记录/teach/LN905 Skill Bank.md"
BASE = "https://neath.clingword.com/api/v1"
START = "<!-- student-os:neath-vocabulary:start -->"
END = "<!-- student-os:neath-vocabulary:end -->"


def load_bank() -> tuple[str, dict]:
    text = BANK.read_text(encoding="utf-8")
    block = text.split(START, 1)[1].split(END, 1)[0]
    match = re.search(r"```json\s*(\{.*\})\s*```", block, re.S)
    if not match:
        raise ValueError("managed vocabulary JSON block is missing")
    data = json.loads(match.group(1))
    collections = {item["name"] for item in data["collections"]}
    seen: set[str] = set()
    for entry in data["entries"]:
        key = entry["word"].strip().casefold()
        if not key or key in seen:
            raise ValueError(f"duplicate or empty vocabulary entry: {entry['word']!r}")
        if entry["collection"] not in collections:
            raise ValueError(f"unknown collection for {entry['word']!r}")
        seen.add(key)
    return text, data


def set_sync_state(text: str, state: str) -> None:
    updated = re.sub(r"(?m)^neath_sync: .*?$", f"neath_sync: {state}", text, count=1)
    if updated != text:
        BANK.write_text(updated, encoding="utf-8")


def request(key: str, path: str, method: str = "GET", body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        BASE + path,
        data=payload,
        method=method,
        headers={"X-Neath-API-Key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def sync(text: str, data: dict) -> tuple[int, int, int]:
    key = subprocess.check_output(
        ["security", "find-generic-password", "-w", "-s", "com.openai.codex.neath-api", "-a", "student-os"],
        text=True,
    ).strip()
    notebooks = {item["name"]: item for item in request(key, "/notebooks")["items"]}
    created = added = updated = 0

    for spec in data["collections"]:
        notebook = notebooks.get(spec["name"])
        if notebook is None:
            notebook = request(key, "/notebooks", "POST", spec)
            notebooks[spec["name"]] = notebook
            created += 1
        notebook_id = notebook["id"]
        current = request(key, f"/notebooks/{notebook_id}/words").get("items", [])
        by_word = {item["word"].strip().casefold(): item for item in current}
        wanted = [entry for entry in data["entries"] if entry["collection"] == spec["name"]]
        missing = []
        for entry in wanted:
            existing = by_word.get(entry["word"].strip().casefold())
            if existing is None:
                missing.append({"word": entry["word"], "sentence": entry["sentence"]})
            elif existing.get("sentence") != entry["sentence"]:
                request(key, f"/words/{existing['id']}", "PATCH", {"sentence": entry["sentence"]})
                updated += 1
        if missing:
            result = request(
                key,
                f"/notebooks/{notebook_id}/words/bulk",
                "POST",
                {"words": missing, "skip_duplicates": True},
            )
            added += int(result.get("added", len(missing)))
            current = request(key, f"/notebooks/{notebook_id}/words").get("items", [])
            by_word = {item["word"].strip().casefold(): item for item in current}
    set_sync_state(text, "synced")
    return created, added, updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate the local managed block only")
    args = parser.parse_args()
    text, data = load_bank()
    if args.check:
        print(f"valid: {len(data['collections'])} collections, {len(data['entries'])} unique entries")
        return 0
    try:
        created, added, updated = sync(text, data)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError, urllib.error.URLError) as exc:
        set_sync_state(text, "pending")
        print(f"sync pending: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"synced: {created} collections created, {added} words added, {updated} words updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
