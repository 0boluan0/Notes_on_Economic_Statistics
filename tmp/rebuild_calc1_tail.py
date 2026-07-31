#!/usr/bin/env python3
"""Replace the oversized final bilingual block in Calculus I with local blocks."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


TRANSLATOR_PATH = Path(__file__).with_name("bilingualize_math_alibaba.py")
SPEC = importlib.util.spec_from_file_location("math_translator", TRANSLATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TRANSLATOR_PATH}")
translator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = translator
SPEC.loader.exec_module(translator)
helper = translator.helper


def collect_tail_units(lines: list[str], start: int) -> list[dict]:
    """Collect short semantic blocks without trusting malformed display-math state."""

    units: list[dict] = []
    index = start
    unit_id = 1
    while index < len(lines):
        line = lines[index]
        heading = helper.HEADING_RE.match(line)
        if heading:
            if helper.CJK_RE.search(helper.visible_text(heading.group(2))):
                units.append(
                    {
                        "id": unit_id,
                        "kind": "heading",
                        "start": index,
                        "end": index,
                        "text": heading.group(2),
                        "quote_prefix": "",
                    }
                )
                unit_id += 1
            index += 1
            continue
        if not line.strip():
            index += 1
            continue

        block_start = index
        block: list[str] = []
        while index < len(lines):
            current = lines[index]
            if index > block_start and helper.HEADING_RE.match(current):
                break
            if not current.strip():
                break
            block.append(current)
            index += 1

        kind = helper.classify(block)
        source = "\n".join(helper.filter_translatable_lines(block, kind))
        if source.strip() and helper.CJK_RE.search(helper.visible_text(source)):
            units.append(
                {
                    "id": unit_id,
                    "kind": kind,
                    "start": block_start,
                    "end": index - 1,
                    "text": source,
                    "quote_prefix": helper.common_quote_prefix(block)
                    if kind in {"callout", "quote"}
                    else "",
                }
            )
            unit_id += 1
    return units


def main() -> None:
    path = Path("01_Math/01_calculus/01_Differentiation.md")
    current = path.read_text()
    last_start = current.rfind(helper.START)
    if last_start < 0:
        raise RuntimeError("final bilingual block not found")
    last_end = current.find(helper.END, last_start)
    if last_end < 0:
        raise RuntimeError("unterminated final bilingual block")
    last_end += len(helper.END)
    if current[last_end : last_end + 1] == "\n":
        last_end += 1
    without_giant = current[:last_start] + current[last_end:]
    lines = without_giant.splitlines()
    tail_start = next(
        index
        for index, line in enumerate(lines)
        if "[!example]- 1E：" in line
    )
    units = collect_tail_units(lines, tail_start)
    translations = translator.request_alibaba_translations(units)
    content = helper.make_content(without_giant, units, translations)
    content = re.sub(
        r"(?m)^[ \t]*(?:>[ \t]*)+(<!-- bilingual-en:(?:start|end) -->)$",
        r"\1",
        content,
    )
    print(helper.patch_for(path, content), end="")


if __name__ == "__main__":
    main()
