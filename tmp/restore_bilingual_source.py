from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path


START_RE = re.compile(r"^(?:>\s*)*<!-- bilingual-en:start -->\s*$")
END_RE = re.compile(r"^(?:>\s*)*<!-- bilingual-en:end -->\s*$")
BLOCK_RE = re.compile(
    r"(?P<lead>\n?)(?P<prefix>^(?:>[ \t]*)*)<!-- bilingual-en:start -->\n"
    r"(?P<body>.*?)\n^(?P=prefix)<!-- bilingual-en:end -->",
    re.MULTILINE | re.DOTALL,
)
WIKILINK_RE = re.compile(r"(!?)\[\[([^\]]+)\]\]")


def visible_text(line: str) -> str:
    def replace(match: re.Match[str]) -> str:
        target, separator, display = match.group(2).partition("|")
        return display if separator else target

    return WIKILINK_RE.sub(replace, line)


def source_line_map(raw_lines: list[str]) -> tuple[list[str], list[int]]:
    source: list[str] = []
    mapping: list[int] = []
    inside = False
    for raw_index, raw_line in enumerate(raw_lines):
        line = raw_line.rstrip("\r\n")
        if START_RE.fullmatch(line):
            if inside:
                raise RuntimeError(f"nested English start at raw line {raw_index + 1}")
            inside = True
            continue
        if END_RE.fullmatch(line):
            if not inside:
                raise RuntimeError(f"orphan English end at raw line {raw_index + 1}")
            inside = False
            continue
        if not inside:
            source.append(line)
            mapping.append(raw_index)
    if inside:
        raise RuntimeError("unclosed English block")
    return source, mapping


def restore(raw: str, baseline: str) -> tuple[str, list[tuple[int, str, str]]]:
    raw_lines = raw.splitlines(keepends=True)
    current, mapping = source_line_map(raw_lines)
    baseline_lines = baseline.splitlines()
    matcher = difflib.SequenceMatcher(
        a=baseline_lines,
        b=current,
        autojunk=False,
    )
    operations = [operation for operation in matcher.get_opcodes() if operation[0] != "equal"]
    retained_links: list[tuple[int, str, str]] = []

    for tag, i1, i2, j1, j2 in reversed(operations):
        baseline_chunk = baseline_lines[i1:i2]
        current_chunk = current[j1:j2]

        if tag == "replace" and len(baseline_chunk) == len(current_chunk):
            for offset, (baseline_line, current_line) in enumerate(
                zip(baseline_chunk, current_chunk)
            ):
                raw_index = mapping[j1 + offset]
                if visible_text(baseline_line) == visible_text(current_line):
                    retained_links.append((i1 + offset + 1, baseline_line, current_line))
                    desired = current_line
                else:
                    desired = baseline_line
                ending = "\n" if raw_lines[raw_index].endswith("\n") else ""
                raw_lines[raw_index] = desired + ending
            continue

        raw_indices = [mapping[index] for index in range(j1, j2)]
        insertion_index = raw_indices[0] if raw_indices else (
            mapping[j1] if j1 < len(mapping) else len(raw_lines)
        )
        for raw_index in reversed(raw_indices):
            del raw_lines[raw_index]
        insertion = [line + "\n" for line in baseline_chunk]
        raw_lines[insertion_index:insertion_index] = insertion

    revised = "".join(raw_lines)

    desired_lines = list(baseline_lines)
    for line_number, _baseline_line, current_line in retained_links:
        desired_lines[line_number - 1] = current_line
    desired = "\n".join(desired_lines)
    if baseline.endswith("\n"):
        desired += "\n"
    stripped = BLOCK_RE.sub("", revised)
    if stripped != desired:
        raise RuntimeError("restored source layer does not match the audited target")
    return revised, sorted(retained_links)


def make_patch(path: Path, original: str, revised: str) -> str:
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            revised.splitlines(keepends=True),
            n=3,
        )
    )[2:]
    if not diff:
        return ""
    hunks: list[str] = []
    for line in diff:
        hunks.append("@@\n" if line.startswith("@@") else line)
    return "".join(
        [
            "*** Begin Patch\n",
            f"*** Update File: {path.resolve()}\n",
            *hunks,
            "*** End Patch\n",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    original = args.path.read_text()
    baseline = args.baseline.read_text()
    revised, retained_links = restore(original, baseline)
    if args.summary:
        print(f"retained_link_migrations={len(retained_links)}")
        for line_number, old, new in retained_links:
            print(f"line {line_number}: {old} -> {new}")
        return
    print(make_patch(args.path, original, revised), end="")


if __name__ == "__main__":
    main()
