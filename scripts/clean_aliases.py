#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean aliases in 00_factor frontmatter:
- keep unique aliases (trimmed, case-insensitive dedupe)
- ensure original Chinese aliases are preserved if present
- remove generic 'hub' alias
- remove aliases identical to filename stem (case-insensitive)
- order: Chinese aliases first, then English

Works without PyYAML: attempts YAML, else string-replaces only aliases block.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "00_factor"

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII_SAFE = re.compile(r"^[A-Za-z0-9 \-\(\)\[\]_\.'+,]+$")


def split_frontmatter(text: str) -> Tuple[Optional[str], str, Optional[str]]:
    if not text.startswith("---\n"):
        return None, text, None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text, None
    fm = text[4:end]
    body = text[end + 5 :]
    return fm, body, text


def parse_yaml_fm(fm_text: str) -> Optional[dict]:
    try:
        import yaml  # type: ignore

        fm = yaml.safe_load(fm_text) or {}
        return fm
    except Exception:
        return None


def build_yaml_fm(fm: dict, body: str) -> Optional[str]:
    try:
        import yaml  # type: ignore

        fm_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return f"---\n{fm_text}---\n{body}"
    except Exception:
        return None


def extract_aliases_from_fm_text(fm_text: str) -> list[str]:
    aliases: list[str] = []
    lines = fm_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("aliases:"):
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("-"):
                item = lines[i].split("-", 1)[1].strip().strip("'\"")
                if item:
                    aliases.append(item)
                i += 1
            break
        i += 1
    return aliases


def replace_aliases_in_fm_text(fm_text: str, new_aliases: list[str]) -> str:
    lines = fm_text.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        if lines[i].strip().startswith("aliases:"):
            out.append("aliases:")
            i += 1
            # skip old list
            while i < len(lines) and lines[i].lstrip().startswith("-"):
                i += 1
            for a in new_aliases:
                out.append(f"- {a}")
            replaced = True
            continue
        out.append(lines[i])
        i += 1
    if not replaced:
        # insert at top
        out.insert(0, "aliases:")
        for a in new_aliases:
            out.insert(1, f"- {a}")
    return "\n".join(out)


def clean_alias_list(stem: str, aliases: list[str]) -> list[str]:
    items: list[str] = []
    seen = set()
    for a in aliases:
        if not isinstance(a, str):
            continue
        s = a.strip()
        if not s:
            continue
        if s.lower() == "hub":
            continue
        key = s.lower()
        if key == stem.lower():
            # remove exact duplicate of filename
            continue
        if key in seen:
            continue
        seen.add(key)
        items.append(s)
    # Order: Chinese first then English
    cn = [x for x in items if CJK.search(x)]
    en = [x for x in items if not CJK.search(x)]
    return cn + en


def process_file(p: Path) -> bool:
    text = p.read_text(encoding="utf-8")
    fm_text, body, full = split_frontmatter(text)
    if fm_text is None:
        return False
    fm = parse_yaml_fm(fm_text)
    changed = False
    stem = p.stem
    if fm is not None:
        aliases = fm.get("aliases")
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []
        new_aliases = clean_alias_list(stem, aliases)
        if new_aliases != aliases:
            fm["aliases"] = new_aliases
            new_doc = build_yaml_fm(fm, body)
            if new_doc:
                p.write_text(new_doc, encoding="utf-8")
                changed = True
            else:
                # fallback replace in fm_text
                new_fm_text = replace_aliases_in_fm_text(fm_text, new_aliases)
                p.write_text(f"---\n{new_fm_text}\n---\n{body}", encoding="utf-8")
                changed = True
    else:
        # no yaml module: string based
        old_aliases = extract_aliases_from_fm_text(fm_text)
        new_aliases = clean_alias_list(stem, old_aliases)
        if new_aliases != old_aliases:
            new_fm_text = replace_aliases_in_fm_text(fm_text, new_aliases)
            p.write_text(f"---\n{new_fm_text}\n---\n{body}", encoding="utf-8")
            changed = True
    return changed


def main() -> int:
    changed = 0
    for p in BASE.rglob("*.md"):
        try:
            if process_file(p):
                changed += 1
        except Exception:
            continue
    print(f"Aliases cleaned in {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

