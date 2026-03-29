#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


ROOTS = [
    "00_factor",
    "01_Math",
    "02_Economy",
    "03_Computer_Science",
    "00_inbox",
    "99_学习情况记录",
    "05_tools",
    "04_Fragments",
]

DEFINITION_HEADINGS = {"定义", "它是什么", "最小可检索信息", "Definition"}
EXAMPLE_HEADINGS = {
    "例题",
    "例子",
    "示例",
    "数值例子",
    "Example",
    "Examples",
    "举例",
    "典型例子",
    "最小例子",
}
INLINE_LABEL_TYPES = {
    "定义": "note",
    "例题": "example",
    "例子": "example",
    "示例": "example",
}

HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
DIRECT_CALL_OUT_RE = re.compile(
    r"^\s*>\s+\[!(note|example)\]([+-])?([^\n]*)$",
    re.IGNORECASE,
)
INLINE_LABEL_RE = re.compile(
    r"^\s*[-*]?\s*(?:\*\*|__)?(定义|例题|例子|示例)(?:\*\*|__)?[：:]\s*(.*)$"
)
DEFINITION_PREFIX_RE = re.compile(
    r"^(定义|它是什么|最小可检索信息|Definition)(?:$|[\s（(：:\-]|[0-9])"
)
EXAMPLE_PREFIX_RE = re.compile(
    r"^(例题|例子|示例|数值例子|Example|Examples|举例|典型例子|最小例子)(?:$|[\s（(：:\-]|[0-9])"
)


@dataclass
class AmbiguousHeading:
    path: Path
    line_no: int
    text: str


@dataclass
class FileResult:
    path: Path
    changed: bool = False
    section_transforms: int = 0
    inline_transforms: int = 0
    canonicalized_callouts: int = 0
    ambiguous: list[AmbiguousHeading] = field(default_factory=list)


@dataclass
class Summary:
    scanned_files: int = 0
    changed_files: int = 0
    section_transforms: int = 0
    inline_transforms: int = 0
    canonicalized_callouts: int = 0
    ambiguous: list[AmbiguousHeading] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize explicit definition/example Markdown blocks into Obsidian callouts."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files in place. Without this flag the script runs in preview mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without rewriting files. This is the default behavior.",
    )
    return parser.parse_args()


def canonicalize_marker(line: str) -> tuple[str, bool]:
    match = DIRECT_CALL_OUT_RE.match(line)
    if not match:
        return line, False

    kind = match.group(1).lower()
    fold_marker = match.group(2) or ""
    suffix = match.group(3) or ""
    new_line = f">[!{kind}]{fold_marker}{suffix}"
    return new_line, new_line != line


def strip_emphasis(text: str) -> str:
    return re.sub(r"(\*\*|__|\*|_)", "", text).strip()


def strip_leading_numbering(text: str) -> str:
    return re.sub(r"^[\s0-9一二三四五六七八九十.．、()（）\-]+", "", text).strip()


def normalize_heading_text(title: str) -> str:
    return strip_leading_numbering(strip_emphasis(title))


def is_blank(line: str) -> bool:
    return not line.strip()


def is_quoted(line: str) -> bool:
    return line.lstrip().startswith(">")


def fence_token(line: str) -> str | None:
    match = FENCE_RE.match(line)
    if not match:
        return None
    return match.group(1)[:3]


def is_heading(line: str) -> tuple[int, str] | None:
    match = HEADING_RE.match(line)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def callout_kind_for_heading(title: str) -> str | None:
    normalized = normalize_heading_text(title)
    if normalized in DEFINITION_HEADINGS:
        return "note"
    if normalized in EXAMPLE_HEADINGS:
        return "example"
    if DEFINITION_PREFIX_RE.match(normalized):
        return "note"
    if EXAMPLE_PREFIX_RE.match(normalized):
        return "example"
    return None


def should_report_ambiguous(title: str) -> bool:
    normalized = normalize_heading_text(title)
    if callout_kind_for_heading(title):
        return False
    return any(token in normalized for token in ("定义", "例题", "例子", "示例", "举例"))


def prefix_callout_body(line: str) -> str:
    if is_blank(line):
        return ">"
    return f"> {line}"


def collect_ambiguous_headings(path: Path, lines: list[str]) -> list[AmbiguousHeading]:
    ambiguous: list[AmbiguousHeading] = []
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    in_fence = False
    current_fence: str | None = None

    for index, line in enumerate(lines, start=1):
        if in_frontmatter:
            if index != 1 and line.strip() == "---":
                in_frontmatter = False
            continue

        if in_fence:
            if line.strip().startswith(current_fence or ""):
                in_fence = False
                current_fence = None
            continue

        if is_quoted(line):
            continue

        token = fence_token(line)
        if token:
            in_fence = True
            current_fence = token
            continue

        heading = is_heading(line)
        if heading and should_report_ambiguous(heading[1]):
            ambiguous.append(AmbiguousHeading(path=path, line_no=index, text=heading[1]))

    return ambiguous


def find_section_end(lines: list[str], start_index: int, level: int) -> int:
    in_fence = False
    current_fence: str | None = None
    index = start_index

    while index < len(lines):
        line = lines[index]
        if in_fence:
            if line.strip().startswith(current_fence or ""):
                in_fence = False
                current_fence = None
            index += 1
            continue

        if is_quoted(line):
            index += 1
            continue

        token = fence_token(line)
        if token:
            in_fence = True
            current_fence = token
            index += 1
            continue

        heading = is_heading(line)
        if heading and heading[0] <= level:
            break

        index += 1

    return index


def find_inline_block_end(lines: list[str], start_index: int) -> int:
    index = start_index
    while index < len(lines):
        line = lines[index]
        if is_blank(line):
            break
        if is_quoted(line):
            break
        if DIRECT_CALL_OUT_RE.match(line):
            break
        if fence_token(line):
            break
        heading = is_heading(line)
        if heading:
            break
        index += 1
    return index


def transform_lines(lines: list[str], result: FileResult) -> list[str]:
    output: list[str] = []
    index = 0
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    in_fence = False
    current_fence: str | None = None

    while index < len(lines):
        line = lines[index]

        if in_frontmatter:
            output.append(line)
            index += 1
            if index > 1 and line.strip() == "---":
                in_frontmatter = False
            continue

        if in_fence:
            output.append(line)
            index += 1
            if line.strip().startswith(current_fence or ""):
                in_fence = False
                current_fence = None
            continue

        canonicalized, changed = canonicalize_marker(line)
        if changed:
            output.append(canonicalized)
            result.canonicalized_callouts += 1
            index += 1
            continue

        if is_quoted(line):
            output.append(line)
            index += 1
            continue

        token = fence_token(line)
        if token:
            output.append(line)
            in_fence = True
            current_fence = token
            index += 1
            continue

        heading = is_heading(line)
        if heading:
            level, title = heading
            kind = callout_kind_for_heading(title)
            if kind:
                section_end = find_section_end(lines, index + 1, level)
                output.append(f">[!{kind}] {title}")
                for body_line in lines[index + 1 : section_end]:
                    output.append(prefix_callout_body(body_line))
                result.section_transforms += 1
                index = section_end
                continue

        inline_match = INLINE_LABEL_RE.match(line)
        if inline_match:
            label = inline_match.group(1)
            kind = INLINE_LABEL_TYPES.get(label)
            if kind:
                block_end = find_inline_block_end(lines, index + 1)
                output.append(f">[!{kind}] {label}")
                remainder = inline_match.group(2)
                if remainder:
                    output.append(prefix_callout_body(remainder))
                for body_line in lines[index + 1 : block_end]:
                    output.append(prefix_callout_body(body_line))
                result.inline_transforms += 1
                index = block_end
                continue

        output.append(line)
        index += 1

    return output


def transform_file(path: Path, apply: bool) -> FileResult:
    result = FileResult(path=path)
    original = path.read_text(encoding="utf-8")
    had_trailing_newline = original.endswith("\n")
    lines = original.splitlines()

    result.ambiguous = collect_ambiguous_headings(path, lines)
    transformed_lines = transform_lines(lines, result)
    new_text = "\n".join(transformed_lines)
    if had_trailing_newline:
        new_text += "\n"

    result.changed = new_text != original
    if apply and result.changed:
        path.write_text(new_text, encoding="utf-8")

    return result


def iter_markdown_files(root_dir: Path) -> list[Path]:
    files: list[Path] = []
    for relative_root in ROOTS:
        base = root_dir / relative_root
        if not base.exists():
            continue
        files.extend(sorted(base.rglob("*.md")))
    return files


def print_summary(summary: Summary, apply: bool) -> None:
    mode = "apply" if apply else "dry-run"
    print(f"Mode: {mode}")
    print(f"Scanned files: {summary.scanned_files}")
    print(f"Changed files: {summary.changed_files}")
    print(f"Transformed heading sections: {summary.section_transforms}")
    print(f"Transformed inline labels: {summary.inline_transforms}")
    print(f"Canonicalized existing note/example callouts: {summary.canonicalized_callouts}")
    print(f"Ambiguous heading candidates: {len(summary.ambiguous)}")

    if summary.ambiguous:
        print("\nAmbiguous heading candidates (manual review):")
        for item in summary.ambiguous:
            print(f"- {item.path}:{item.line_no}: {item.text}")


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    files = iter_markdown_files(root_dir)

    summary = Summary()
    summary.scanned_files = len(files)

    for path in files:
        result = transform_file(path, apply=args.apply)

        if result.changed:
            summary.changed_files += 1
        summary.section_transforms += result.section_transforms
        summary.inline_transforms += result.inline_transforms
        summary.canonicalized_callouts += result.canonicalized_callouts
        summary.ambiguous.extend(result.ambiguous)

    print_summary(summary, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
