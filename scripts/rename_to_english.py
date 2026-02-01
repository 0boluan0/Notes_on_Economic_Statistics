#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename 00_factor Markdown files to English filenames when possible.
- Picks an English alias from frontmatter `aliases` (ASCII letters/digits/spaces/hyphen/parentheses)
- Preserves Chinese (original) name by ensuring it exists in `aliases`
- Keeps '-hub' suffix if the original filename had it (e.g., 'Growth Theory-hub.md')
- Skips if no suitable English alias is found or name already English

Safe behavior:
- Only processes files under 00_factor
- Writes updated frontmatter when adding missing alias
- Prints a summary of planned and applied changes
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "00_factor"

ASCII_SAFE = re.compile(r"^[A-Za-z0-9 \-\(\)\[\]_\.'+,]+$")


def parse_frontmatter(text: str) -> Tuple[dict, str] | Tuple[None, str]:
    if not text.startswith("---\n"):
        return None, text
    try:
        end = text.find("\n---\n", 4)
        if end == -1:
            return None, text
        fm_text = text[4:end]
        body = text[end + 5 :]
        try:
            import yaml  # type: ignore

            fm = yaml.safe_load(fm_text) or {}
        except Exception:
            # Fallback: very simple parser for aliases only
            fm = {}
            aliases: list[str] = []
            lines = fm_text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].rstrip()
                if line.strip().startswith("aliases:"):
                    # collect subsequent list items
                    i += 1
                    while i < len(lines) and lines[i].lstrip().startswith("-"):
                        item = lines[i].split("-", 1)[1].strip()
                        # strip surrounding quotes if any
                        item = item.strip('"')
                        item = item.strip("'")
                        if item:
                            aliases.append(item)
                        i += 1
                    fm["aliases"] = aliases
                    continue
                i += 1
        return fm, body
    except Exception:
        return None, text


def dump_frontmatter(fm: dict, body: str) -> Optional[str]:
    try:
        import yaml  # type: ignore

        fm_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return f"---\n{fm_text}---\n{body}"
    except Exception:
        return None


def is_ascii_name(name: str) -> bool:
    return bool(ASCII_SAFE.match(name))


def choose_english_alias(aliases: list[str], want_hub: bool) -> Optional[str]:
    cand = None
    for a in aliases:
        if not isinstance(a, str):
            continue
        a_stripped = a.strip()
        if is_ascii_name(a_stripped) and a_stripped:
            # avoid generic 'hub' alone
            if want_hub and a_stripped.lower() == "hub":
                continue
            cand = a_stripped
            break
    return cand


def sanitize_filename(name: str) -> str:
    # Replace slashes/colons, collapse spaces
    name = name.replace("/", "-").replace(":", "-")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def main(apply: bool = True) -> int:
    TRANSLATE = {
        # framework
        '帕累托': 'Pareto',
        # system
        '资本金持有率': 'Capital Holding Ratio',
        '库克比率': "Cook's Ratio",
        # procedure
        '相关系数': 'Correlation Coefficient',
        # concept
        '单因子模型': 'Single-Factor Model',
        '已实现波动率': 'Realized Volatility',
        '线性产品': 'Linear Products',
        '修正久期': 'Modified Duration',
        '残差': 'Residual',
        '阿尔法值': 'Alpha',
        '消费者剩余': 'Consumer Surplus',
        '隐含波动率': 'Implied Volatility',
        '契约曲线': 'Contract Curve',
        '大样本与小样本': 'Large and Small Samples',
        '曲率': 'Curvature',
        '非线性产品': 'Nonlinear Products',
        '等待时间分布': 'Waiting Time Distribution',
        '生产可能性曲线': 'Production Possibility Frontier (PPF)',
        '收益率曲线风险': 'Yield Curve Risk',
        '更新过程': 'Renewal Process',
        '隐含期权风险': 'Implied Option Risk',
        '独立性与不相关': 'Independence vs. Uncorrelated',
        '两个随机变量线性组合': 'Linear Combination of Two Random Variables',
        '埃奇沃斯框图': 'Edgeworth Box',
        '净现值': 'Net Present Value',
        '资本市场线': 'Capital Market Line',
        '有效久期': 'Effective Duration',
        '基差风险': 'Basis Risk',
        '历史波动率': 'Historical Volatility',
        '局部久期': 'Key Rate Duration',
        '收入效应与替代效应': 'Income and Substitution Effects',
        '多因子模型': 'Multi-Factor Model',
        '非齐次泊松过程': 'Nonhomogeneous Poisson Process',
        '两部门模型': 'Two-Sector Model',
    }
    planned = []
    for md in BASE.rglob("*.md"):
        rel = md.relative_to(ROOT)
        stem = md.stem
        want_hub = stem.endswith("-hub")
        # Skip if already mostly ASCII (filename base)
        if is_ascii_name(stem):
            # still ensure original stem alias tracked; nothing to rename
            # but if contains spaces only ASCII okay
            pass

        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, body = parse_frontmatter(text)
        if fm is None:
            continue
        aliases = fm.get("aliases")
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []

        # ensure original stem alias（若无法写回，将跳过更新）
        need_write = False
        if stem not in aliases:
            aliases.append(stem)
            need_write = True

        new_base = None
        if not is_ascii_name(stem):
            eng = choose_english_alias(aliases, want_hub)
            if eng:
                eng = sanitize_filename(eng)
                if want_hub and not eng.endswith("-hub"):
                    new_base = f"{eng}-hub"
                else:
                    new_base = eng
            # Debug: print if no english alias found
            else:
                # try static mapping by stem
                mapped = TRANSLATE.get(stem)
                if mapped:
                    mapped = sanitize_filename(mapped)
                    if mapped not in aliases:
                        aliases.append(mapped)
                        need_write = True
                    if want_hub and not mapped.endswith('-hub'):
                        new_base = f"{mapped}-hub"
                    else:
                        new_base = mapped
                else:
                    print(f"NO_ENGLISH_ALIAS for {rel} aliases={aliases}")

        # Write back aliases if changed
        if need_write:
            fm["aliases"] = aliases
            new_text = dump_frontmatter(fm, body)
            if new_text and new_text != text:
                if apply:
                    md.write_text(new_text, encoding="utf-8")

        if new_base and new_base != stem:
            new_name = new_base + ".md"
            new_path = md.with_name(new_name)
            if not new_path.exists():
                planned.append((md, new_path))

    # Apply renames
    for old, new in planned:
        if apply:
            new = Path(new)
            old.rename(new)
        print(f"RENAMED: {old.relative_to(ROOT)} -> {new.relative_to(ROOT)}")

    print(f"Total renamed: {len(planned)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(True))
