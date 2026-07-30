#!/usr/bin/env python3
"""Generate an apply_patch payload that only inserts bilingual English layers."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path


CJK_RE = re.compile(r"[\u3400-\u9fff]")
CHINESE_PUNCT_RE = re.compile(r"[，。；：、！？（）《》【】]")
START = "<!-- bilingual-en:start -->"
END = "<!-- bilingual-en:end -->"
WIKILINK_RE = re.compile(r"!?\[\[([^\]]+)\]\]")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
QUOTE_RE = re.compile(r"^(\s*(?:>\s*)+)(.*)$")
FENCE_RE = re.compile(r"^(?:```|~~~)")


def strip_quote(line: str) -> str:
    match = QUOTE_RE.match(line)
    return match.group(2) if match else line


def structural_core(line: str) -> str:
    """Remove quote/list indentation only for detecting fences and math."""
    line = strip_quote(line).lstrip()
    line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
    return line.strip()


def visible_text(text: str) -> str:
    """Approximate Obsidian reading-view text for the no-CJK check."""

    def wiki(match: re.Match[str]) -> str:
        spec = match.group(1)
        if match.group(0).startswith("!"):
            return ""
        if "|" in spec:
            return spec.split("|", 1)[1]
        return spec.split("#", 1)[0]

    text = WIKILINK_RE.sub(wiki, text)
    text = MD_LINK_RE.sub(lambda m: m.group(1), text)
    text = INLINE_CODE_RE.sub("", text)
    text = re.sub(r"https?://\S+", "", text)
    return text


def filter_translatable_lines(lines: list[str], kind: str) -> list[str]:
    """Remove code/math payload and quote prefixes before translation."""
    filtered: list[str] = []
    in_fence = False
    in_math = False
    for index, raw in enumerate(lines):
        line = strip_quote(raw)
        core = structural_core(raw)
        if FENCE_RE.match(core):
            in_fence = not in_fence
            continue
        if core == "$$":
            in_math = not in_math
            continue
        if in_fence or in_math:
            continue
        if re.fullmatch(r"\s*\^[A-Za-z0-9_-]+\s*", line):
            continue
        if kind == "callout" and index == 0 and re.match(r"^\[![^\]]+\]", line):
            title = re.sub(r"^\[![^\]]+\][+-]?\s*", "", line).strip()
            if CJK_RE.search(visible_text(title)):
                filtered.append(f"**{title}**")
            continue
        filtered.append(line)
    while filtered and not filtered[0].strip():
        filtered.pop(0)
    while filtered and not filtered[-1].strip():
        filtered.pop()
    return filtered


def classify(lines: list[str]) -> str:
    first = next((line for line in lines if line.strip()), "")
    plain = strip_quote(first).lstrip()
    if re.match(r"^\[![^\]]+\]", plain):
        return "callout"
    if plain.startswith("|"):
        return "table"
    if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", plain):
        return "list"
    if QUOTE_RE.match(first):
        return "quote"
    return "paragraph"


def common_quote_prefix(lines: list[str]) -> str:
    prefixes = []
    for line in lines:
        if not line.strip():
            continue
        match = QUOTE_RE.match(line)
        if not match:
            return ""
        prefixes.append(match.group(1))
    return min(prefixes, key=len) if prefixes else ""


def collect_units(lines: list[str]) -> list[dict]:
    units: list[dict] = []
    index = 0
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    if in_frontmatter:
        index = 1
        while index < len(lines):
            if lines[index].strip() == "---":
                index += 1
                break
            index += 1

    unit_id = 1
    while index < len(lines):
        line = lines[index]
        heading = HEADING_RE.match(line)
        if heading:
            if CJK_RE.search(visible_text(heading.group(2))):
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

        core = structural_core(line)
        if FENCE_RE.match(core):
            fence_token = core[:3]
            index += 1
            while index < len(lines):
                if structural_core(lines[index]).startswith(fence_token):
                    index += 1
                    break
                index += 1
            continue
        if core == "$$":
            index += 1
            while index < len(lines):
                if structural_core(lines[index]) == "$$":
                    index += 1
                    break
                index += 1
            continue
        if not line.strip():
            index += 1
            continue

        start = index
        block: list[str] = []
        nested_fence = False
        nested_math = False
        while index < len(lines):
            current = lines[index]
            if index > start and HEADING_RE.match(current) and not nested_fence and not nested_math:
                break
            current_core = structural_core(current)
            if FENCE_RE.match(current_core):
                nested_fence = not nested_fence
            elif current_core == "$$" and not nested_fence:
                nested_math = not nested_math
            if not current.strip() and not nested_fence and not nested_math:
                break
            block.append(current)
            index += 1

        kind = classify(block)
        translatable = filter_translatable_lines(block, kind)
        source = "\n".join(translatable)
        if source.strip() and CJK_RE.search(visible_text(source)):
            units.append(
                {
                    "id": unit_id,
                    "kind": kind,
                    "start": start,
                    "end": index - 1,
                    "text": source,
                    "quote_prefix": common_quote_prefix(block)
                    if kind in {"callout", "quote"}
                    else "",
                }
            )
            unit_id += 1
    return units


SYSTEM_PROMPT = """You are a meticulous Chinese-to-English translator for introductory computer-science lecture notes. Return JSON only.

For every input unit, provide one faithful, idiomatic English Markdown translation with the same id and kind.
- Translate every visible Chinese statement; do not summarize, omit, add commentary, or leave Chinese visible.
- Preserve the meaning and order. Keep existing English technical terms when they are already natural, and use standard computer-science terminology.
- Preserve inline-code spans, formulas, URLs, Markdown emphasis, list/checklist/table structure, and link destinations.
- For a wikilink whose destination contains Chinese, keep that destination but provide an idiomatic English display label: [[中文目标|English label]].
- Do not output bilingual markers, Markdown headings, callout declarations, code fences, YAML, or block IDs.
- A heading translation is plain title text; the caller will add italics.
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "kind": {"type": "string"},
                    "english": {"type": "string"},
                },
                "required": ["id", "kind", "english"],
            },
        }
    },
    "required": ["translations"],
}


def list_item_count(text: str) -> int:
    return sum(
        bool(re.match(r"^\s*(?:[-*+]\s+(?:\[[ xX]\]\s+)?|\d+[.)]\s+)", line))
        for line in text.splitlines()
    )


def preserved_tokens(text: str) -> dict[str, list[str]]:
    wiki_targets = []
    for match in WIKILINK_RE.finditer(text):
        spec = match.group(1)
        wiki_targets.append(spec.split("|", 1)[0])
    markdown_targets = re.findall(r"!?\[[^\]]*\]\(([^)]*)\)", text)
    return {
        "code": sorted(INLINE_CODE_RE.findall(text)),
        "math": sorted(re.findall(r"(?<!\$)\$[^$\n]+\$(?!\$)", text)),
        "wiki": sorted(wiki_targets),
        "markdown": sorted(markdown_targets),
    }


def translation_issues(unit: dict, english: str) -> list[str]:
    english = normalize_english(english, unit["kind"])
    issues: list[str] = []
    visible = visible_text(english)
    if not english:
        issues.append("empty translation")
    if CJK_RE.search(visible):
        issues.append("visible Chinese characters remain")
    if CHINESE_PUNCT_RE.search(visible):
        issues.append("Chinese punctuation remains")
    source_items = 0 if unit["kind"] == "heading" else list_item_count(unit["text"])
    english_items = 0 if unit["kind"] == "heading" else list_item_count(english)
    if source_items != english_items:
        issues.append(
            f"list-item count differs: source={source_items}, translation={english_items}"
        )
    source_rows = sum(line.lstrip().startswith("|") for line in unit["text"].splitlines())
    english_rows = sum(line.lstrip().startswith("|") for line in english.splitlines())
    if source_rows != english_rows:
        issues.append(
            f"table-row count differs: source={source_rows}, translation={english_rows}"
        )
    if unit["kind"] == "heading":
        source_number = re.match(r"^(\d+)\.", unit["text"])
        english_number = re.match(r"^(\d+)\.", english)
        if bool(source_number) != bool(english_number) or (
            source_number
            and english_number
            and source_number.group(1) != english_number.group(1)
        ):
            issues.append("heading number was not preserved")
        if "\n" in english:
            issues.append("heading translation spans multiple lines")
    source_tokens = preserved_tokens(unit["text"])
    english_tokens = preserved_tokens(english)
    for token_kind in source_tokens:
        if token_kind == "code":
            changed = bool(
                Counter(source_tokens[token_kind]) - Counter(english_tokens[token_kind])
            )
        else:
            changed = source_tokens[token_kind] != english_tokens[token_kind]
        if changed:
            issues.append(f"{token_kind} tokens or destinations changed")
    return issues


def request_translation_batch(
    units: list[dict],
    model: str,
    previous: dict[int, str] | None = None,
    issue_map: dict[int, list[str]] | None = None,
) -> dict[int, str]:
    request_units = [
        {
            "id": unit["id"],
            "kind": unit["kind"],
            "text": unit["text"],
            "expected_list_items": 0
            if unit["kind"] == "heading"
            else list_item_count(unit["text"]),
            **(
                {
                    "previous_draft": previous.get(unit["id"], ""),
                    "validation_errors": (issue_map or {}).get(unit["id"], []),
                }
                if previous is not None
                else {}
            ),
        }
        for unit in units
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": (
                            "Translate these units. Match every bullet and sentence."
                            if previous is None
                            else "Revise every previous draft to fix all listed validation errors. "
                            "Return complete, idiomatic English translations, not patches."
                        ),
                        "units": request_units,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": 32768},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    parsed = json.loads(result["message"]["content"])
    translated = parsed.get("translations", [])
    by_id = {item["id"]: item["english"].strip() for item in translated}
    expected = {unit["id"] for unit in units}
    extra = sorted(set(by_id) - expected)
    if extra:
        raise RuntimeError(f"translation ids mismatch: extra={extra}")
    return by_id


def request_translations(units: list[dict], model: str) -> dict[int, str]:
    by_id: dict[int, str] = {}
    batch_size = 10
    for start in range(0, len(units), batch_size):
        batch = units[start : start + batch_size]
        batch_result = request_translation_batch(batch, model)
        by_id.update(batch_result)
        missing = [unit for unit in batch if unit["id"] not in batch_result]
        for _ in range(3):
            if not missing:
                break
            retry_result = request_translation_batch(missing, model)
            by_id.update(retry_result)
            missing = [unit for unit in missing if unit["id"] not in retry_result]
        if missing:
            raise RuntimeError(
                f"translation ids still missing after retries: "
                f"{[unit['id'] for unit in missing]}"
            )
        issue_map = {
            unit["id"]: translation_issues(unit, by_id[unit["id"]])
            for unit in batch
        }
        invalid = [unit for unit in batch if issue_map[unit["id"]]]
        for _ in range(3):
            if not invalid:
                break
            previous = {unit["id"]: by_id.get(unit["id"], "") for unit in invalid}
            retry_issues = {unit["id"]: issue_map[unit["id"]] for unit in invalid}
            retry_result = request_translation_batch(
                invalid, model, previous=previous, issue_map=retry_issues
            )
            by_id.update(retry_result)
            issue_map.update(
                {
                    unit["id"]: translation_issues(
                        unit, by_id.get(unit["id"], "")
                    )
                    for unit in invalid
                }
            )
            invalid = [unit for unit in invalid if issue_map[unit["id"]]]
        if invalid:
            raise RuntimeError(
                "translation validation still fails after retries: "
                + repr(
                    {
                        unit["id"]: issue_map[unit["id"]]
                        for unit in invalid
                    }
                )
            )
    return by_id


def normalize_english(text: str, kind: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.replace(START, "").replace(END, "").strip()
    if kind == "heading":
        text = re.sub(r"^#{1,6}\s+", "", text).strip()
        text = text.strip("*_ ")
        if "\n" in text:
            text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    return text


def visible_english_has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(visible_text(text)))


def build_insertion(unit: dict, english: str) -> list[str]:
    kind = unit["kind"]
    english = normalize_english(english, kind)
    if not english:
        raise RuntimeError(f"empty translation for unit {unit['id']}")
    if visible_english_has_cjk(english):
        raise RuntimeError(
            f"visible CJK remains in unit {unit['id']}: {english[:240]!r}"
        )
    if kind == "heading":
        body = [f"*{english}*"]
        prefix = ""
    else:
        body = english.splitlines()
        prefix = unit["quote_prefix"]
    if prefix:
        rendered = [prefix + START]
        rendered.extend(prefix.rstrip() if not line else prefix + line for line in body)
        rendered.append(prefix + END)
        return rendered
    return [START, *body, END]


def make_content(original: str, units: list[dict], translations: dict[int, str]) -> str:
    final_newline = original.endswith("\n")
    lines = original.splitlines()
    after: dict[int, list[str]] = {}
    for unit in units:
        if unit["end"] in after:
            raise RuntimeError(f"overlapping insertion at line {unit['end'] + 1}")
        after[unit["end"]] = build_insertion(unit, translations[unit["id"]])
    result: list[str] = []
    for index, line in enumerate(lines):
        result.append(line)
        if index in after:
            result.extend(after[index])
    output = "\n".join(result)
    if final_newline:
        output += "\n"
    return output


def patch_for(path: Path, content: str) -> str:
    absolute = path.resolve()
    additions = "".join(f"+{line}\n" for line in content.splitlines())
    if content and not content.endswith("\n"):
        additions = additions.rstrip("\n") + "\n"
    return (
        "*** Begin Patch\n"
        f"*** Delete File: {absolute}\n"
        f"*** Add File: {absolute}\n"
        f"{additions}"
        "*** End Patch\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", default="deepseek-r1:14b")
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()

    path = args.path
    baseline = Path("tmp/bilingual_baseline/classroom") / path
    if not baseline.exists():
        raise SystemExit(f"missing baseline: {baseline}")
    original = baseline.read_text()
    current = path.read_text()
    if START not in current and current != original:
        raise SystemExit(f"current file differs from baseline before bilingualization: {path}")
    units = collect_units(original.splitlines())
    if args.inventory:
        print(json.dumps(units, ensure_ascii=False, indent=2))
        return
    translations = request_translations(units, args.model)
    content = make_content(original, units, translations)
    print(patch_for(path, content), end="")


if __name__ == "__main__":
    main()
