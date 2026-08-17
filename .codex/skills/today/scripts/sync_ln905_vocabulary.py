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
        if not entry.get("meaning", "").strip() or not entry.get("sentence", "").strip():
            raise ValueError(f"meaning or sentence missing for {entry['word']!r}")
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


def managed_definition(entry: dict) -> dict:
    return {
        "word": entry["word"],
        "mode": "en_to_zh",
        "exists": True,
        "phonetic": "",
        "origin": "student-os",
        "meanings": [
            {
                "part_of_speech": "phrase" if " " in entry["word"].strip() else "term",
                "definition": entry["meaning"],
                "examples": [],
            }
        ],
        "variations": [],
        "synonyms": [],
        "antonyms": [],
    }


def has_definition(definition: object) -> bool:
    return bool(
        isinstance(definition, dict)
        and definition.get("exists")
        and any(item.get("definition", "").strip() for item in definition.get("meanings", []))
    )


def definition_needs_update(definition: object, entry: dict) -> bool:
    if not has_definition(definition):
        return True
    if not isinstance(definition, dict) or definition.get("origin") != "student-os":
        return False
    return definition.get("meanings", [{}])[0].get("definition") != entry["meaning"]


def get_key() -> str:
    return subprocess.check_output(
        ["security", "find-generic-password", "-w", "-s", "com.openai.codex.neath-api", "-a", "student-os"],
        text=True,
    ).strip()


def remote_state(key: str, data: dict) -> tuple[dict, dict]:
    notebooks = {item["name"]: item for item in request(key, "/notebooks")["items"]}
    words = {
        name: request(key, f"/notebooks/{notebooks[name]['id']}/words?limit=500").get("items", [])
        for name in (item["name"] for item in data["collections"])
        if name in notebooks
    }
    return notebooks, words


def check_remote(data: dict) -> None:
    key = get_key()
    notebooks, remote_words = remote_state(key, data)
    missing_notebooks = [item["name"] for item in data["collections"] if item["name"] not in notebooks]
    if missing_notebooks:
        raise ValueError(f"missing Neath collections: {missing_notebooks}")
    checked = 0
    for spec in data["collections"]:
        wanted = [item for item in data["entries"] if item["collection"] == spec["name"]]
        wanted_keys = {item["word"].strip().casefold() for item in wanted}
        extras = sorted(
            item["word"]
            for item in remote_words[spec["name"]]
            if item["word"].strip().casefold() not in wanted_keys
        )
        if extras:
            raise ValueError(f"unmanaged words remain in {spec['name']}: {extras}")
        by_word = {item["word"].strip().casefold(): item for item in remote_words[spec["name"]]}
        for entry in wanted:
            remote = by_word.get(entry["word"].strip().casefold())
            if remote is None:
                raise ValueError(f"word missing from Neath: {entry['word']!r}")
            if remote.get("sentence") != entry["sentence"]:
                raise ValueError(f"sentence mismatch in Neath: {entry['word']!r}")
            if not has_definition(remote.get("definition")):
                raise ValueError(f"definition missing from Neath: {entry['word']!r}")
            checked += 1
        print(f"recognised: {spec['name']}")
    print(f"remote definitions valid: {checked}/{len(data['entries'])}")


def sync(text: str, data: dict) -> tuple[int, int, int, int]:
    key = get_key()
    notebooks, remote_words = remote_state(key, data)
    created = added = updated = defined = 0

    for spec in data["collections"]:
        notebook = notebooks.get(spec["name"])
        if notebook is None:
            notebook = request(key, "/notebooks", "POST", spec)
            notebooks[spec["name"]] = notebook
            created += 1
        notebook_id = notebook["id"]
        current = remote_words.get(spec["name"], [])
        by_word = {item["word"].strip().casefold(): item for item in current}
        wanted = [entry for entry in data["entries"] if entry["collection"] == spec["name"]]
        missing = []
        for entry in wanted:
            existing = by_word.get(entry["word"].strip().casefold())
            if existing is None:
                missing.append(
                    {
                        "word": entry["word"],
                        "sentence": entry["sentence"],
                        "definition": managed_definition(entry),
                    }
                )
                continue
            changes = {}
            if existing.get("sentence") != entry["sentence"]:
                changes["sentence"] = entry["sentence"]
            if definition_needs_update(existing.get("definition"), entry):
                changes["definition"] = managed_definition(entry)
                defined += 1
            if changes:
                request(key, f"/words/{existing['id']}", "PATCH", changes)
                updated += 1
        if missing:
            result = request(
                key,
                f"/notebooks/{notebook_id}/words/bulk",
                "POST",
                {"words": missing, "skip_duplicates": True},
            )
            added += int(result.get("added", len(missing)))
            defined += len(missing)
    set_sync_state(text, "synced")
    return created, added, updated, defined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate the local managed block only")
    parser.add_argument("--check-remote", action="store_true", help="validate every managed Neath card and definition")
    args = parser.parse_args()
    text, data = load_bank()
    if args.check:
        print(f"valid: {len(data['collections'])} collections, {len(data['entries'])} unique entries")
        return 0
    if args.check_remote:
        check_remote(data)
        return 0
    try:
        created, added, updated, defined = sync(text, data)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError, urllib.error.URLError) as exc:
        set_sync_state(text, "pending")
        print(f"sync pending: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        f"synced: {created} collections created, {added} words added, "
        f"{updated} words updated, {defined} definitions written"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
