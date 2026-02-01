#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit 01_Math and 02_Economy notes for potential KCS knowledge points
and check corresponding cards and backlinks in 00_factor.

Heuristics:
- Candidates from headings starting with '#', '##', '###'
- Clean generic headings (e.g., 引言/导论/介绍/课程笔记/总结)
- For each candidate, check if a 00_factor card exists with the same filename
  or mentions the candidate in aliases/frontmatter or title line.
- Then check if the source note contains a wiki link like [[Candidate]]

Outputs a concise text report to stdout.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
NOTES_DIRS = [ROOT/"01_Math", ROOT/"02_Economy"]
CARDS_DIR = ROOT/"00_factor"

GENERIC_HEADINGS = {
    "引言", "导论", "介绍", "introduction", "课程笔记", "总结", "考试内容", "复习",
    "回忆用", "目录", "前言", "preface", "perface", "outline"
}

KEEP_KEYWORDS = [
    # Chinese domain terms likely to be concepts
    r"模型", r"检验", r"定理", r"方程", r"分布", r"过程", r"函数", r"矩阵", r"协方差", r"相关",
    r"平稳", r"协整", r"单位根", r"波动", r"估计", r"回归", r"最小二乘", r"似然", r"期望",
    r"VaR", r"ES", r"CVaR", r"风险", r"久期", r"凸性", r"利率", r"期权", r"希腊",
    r"纳什", r"均衡", r"贝叶斯", r"马尔可夫", r"泊松", r"布朗", r"鞅",
    # English signals
    r"ARMA", r"ARIMA", r"VAR", r"GARCH", r"ARCH", r"ADF", r"KPSS", r"Johansen",
    r"OLS", r"MLE", r"CLT", r"LLN", r"PCA", r"EMH",
]

IGNORE_KEYWORDS = [
    r"作业", r"案例", r"举例", r"原理", r"应用", r"考试", r"习题", r"练习", r"总结",
    r"扩展", r"引言", r"导论", r"介绍", r"概览", r"回顾", r"作图", r"证明题",
]

def iter_md_files(base: Path):
    for p in base.rglob("*.md"):
        # exclude hidden and .obsidian
        parts = set(p.parts)
        if any(s.startswith('.') for s in p.parts):
            continue
        yield p


def extract_candidates(path: Path) -> list[tuple[str,int]]:
    cands: list[tuple[str,int]] = []
    try:
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                m = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
                if not m:
                    continue
                title = m.group(2)
                # strip leading numbering like '1. ', '01_', '（一）', '1 ' etc.
                title = re.sub(r"^[0-9]+[\.、_\-\s]*", "", title)
                title = re.sub(r"^[\(（\[]?[0-9一二三四五六七八九十]+[\)）\]]\s*", "", title)
                # drop trailing punctuation
                title = title.strip().strip('#').strip()
                if not title:
                    continue
                # filter generic headings (case-insensitive contains)
                tnorm = title.lower()
                if any(g in tnorm for g in [g.lower() for g in GENERIC_HEADINGS]):
                    continue
                if any(re.search(p, title) for p in KEEP_KEYWORDS) and not any(re.search(p, title) for p in IGNORE_KEYWORDS):
                    cands.append((title, i))
    except Exception:
        pass
    return cands


def has_card(candidate: str) -> bool:
    # Exact filename match by iterating to avoid glob pattern issues
    for p in CARDS_DIR.rglob("*.md"):
        try:
            if p.stem == candidate:
                return True
        except Exception:
            pass
    # Search in aliases or title lines
    pat = re.escape(candidate)
    try:
        import subprocess
        res = subprocess.run([
            "rg", "-n", "-S", pat, str(CARDS_DIR)
        ], capture_output=True, text=True)
        return bool(res.stdout.strip())
    except Exception:
        # fallback: linear scan could be expensive; assume not found
        return False


def has_backlink(note_path: Path, candidate: str) -> bool:
    # naive: check an exact [[candidate]] occurrence anywhere in file
    try:
        txt = note_path.read_text(encoding='utf-8')
    except Exception:
        return False
    needle1 = f"[[{candidate}]]"
    needle2 = f"[[{candidate}#"
    return (needle1 in txt) or (needle2 in txt)


def main() -> int:
    report_missing_card = []   # (note, line, candidate)
    report_missing_backlink = []  # (note, line, candidate)
    per_note_counts = defaultdict(lambda: {"cands":0, "missing_card":0, "missing_link":0})

    for ndir in NOTES_DIRS:
        for md in iter_md_files(ndir):
            cands = extract_candidates(md)
            per_note_counts[str(md)]["cands"] += len(cands)
            for cand, ln in cands:
                card_exists = has_card(cand)
                if not card_exists:
                    report_missing_card.append((str(md), ln, cand))
                    per_note_counts[str(md)]["missing_card"] += 1
                else:
                    if not has_backlink(md, cand):
                        report_missing_backlink.append((str(md), ln, cand))
                        per_note_counts[str(md)]["missing_link"] += 1

    # Summarize
    total_notes = len(per_note_counts)
    total_cands = sum(v["cands"] for v in per_note_counts.values())
    total_missing_cards = len(report_missing_card)
    total_missing_links = len(report_missing_backlink)

    print("KCS Audit Summary")
    print(f"Notes scanned: {total_notes}")
    print(f"Candidate headings: {total_cands}")
    print(f"Missing cards: {total_missing_cards}")
    print(f"Missing backlinks: {total_missing_links}")
    print()

    def preview(rows, title, limit=25):
        if not rows:
            print(f"{title}: None")
            print()
            return
        print(f"{title}: (showing up to {limit})")
        for i, (p,l,c) in enumerate(rows[:limit], start=1):
            rel = str(Path(p).relative_to(ROOT))
            print(f"{i:2d}. {rel}:{l} -> {c}")
        print()

    preview(report_missing_card, "Missing card candidates")
    preview(report_missing_backlink, "Existing card but missing backlink")

    # Top problem notes
    worst = sorted(per_note_counts.items(), key=lambda kv: (kv[1]["missing_card"]+kv[1]["missing_link"]))[::-1][:10]
    print("Top notes needing attention:")
    for (p, v) in worst:
        rel = str(Path(p).relative_to(ROOT))
        print(f"- {rel}  (cands={v['cands']}, missing_card={v['missing_card']}, missing_link={v['missing_link']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
