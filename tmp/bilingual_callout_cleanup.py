from __future__ import annotations

import argparse
import difflib
import hashlib
import re
from pathlib import Path


BLOCK_RE = re.compile(
    r"(?P<lead>\n?)(?P<prefix>^(?:>[ \t]*)*)<!-- bilingual-en:start -->\n"
    r"(?P<body>.*?)\n^(?P=prefix)<!-- bilingual-en:end -->",
    re.MULTILINE | re.DOTALL,
)

NAV_EN = (
    "course navigation",
    "quick navigation",
    "local material",
    "materials & exercises",
    "materials and exercises",
    "table of contents",
    "source index",
    "material index",
    "official scope and materials",
    "full exam paper and official solutions",
)

NAV_EN_TITLES = {
    "course navigation",
    "course catalog",
    "quick navigation",
    "local material",
    "local materials",
    "local materials & exercises",
    "local materials and exercises",
    "materials & exercises",
    "materials and exercises",
    "table of contents",
    "contents",
    "source index",
    "material index",
    "official scope and materials",
    "full exam paper and official solutions",
    "complete official papers and answers",
}

NAV_ZH = (
    "课程导航",
    "快速导航",
    "本地材料",
    "材料与练习",
    "本地材料与练习",
    "目录",
    "材料索引",
    "资料索引",
    "原卷与官方解答",
)


def unquote(line: str) -> str:
    return re.sub(r"^[ \t]*(?:>[ \t]?)+", "", line)


def drop_duplicate_callout_title(lines: list[str]) -> tuple[list[str], bool]:
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return lines, False

    visible = unquote(lines[first]).strip()
    emphasized = re.fullmatch(
        r"(?:\*\*|__|\*)(.+?)(?:\*\*|__|\*)", visible
    )
    duplicate = visible.startswith("[!")
    if emphasized:
        title = emphasized.group(1).strip().lower()
        generic_title = bool(
            re.search(
                r"(?:self[- ]?test|self[- ]?check|questions?.*answers?|"
                r"common (?:pitfall|error)|main line|official|checklist|"
                r"strict instruction|applicable condition|\btheorem\b|"
                r"\brule\b|power law|general structure|geometric intuition|"
                r"errata|lowest acceptance|read it|why can(?:not|'t)|"
                r"continuity at a point|commercial code|natural exponential function|"
                r"natural logarithmic derivative|basic limits? of|problem set .*errors?)",
                title,
            )
            or bool(re.match(r"^\d+[a-z]:\s", title))
        )
        duplicate = duplicate or generic_title
    if not duplicate:
        return lines, False
    return lines[:first] + lines[first + 1 :], True


def is_link_structure(line: str) -> bool:
    line = line.strip()
    if line.startswith("|"):
        return True

    line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
    line = re.sub(r"^\[[ xX]\]\s+", "", line)
    line = re.sub(
        r"^(?:main note|official lesson|official solutions?|source|answers?|solutions?)\s*:\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    if "[[" not in line and "](" not in line and not re.search(r"https?://", line):
        return False

    residue = re.sub(r"!?\[\[[^\]]+\]\]", "", line)
    residue = re.sub(r"\[[^\]]*\]\([^)]*\)", "", residue)
    residue = re.sub(r"https?://\S+", "", residue)
    residue = re.sub(r"[·|,;:，；：()（）\s✅0-9./–—-]+", "", residue)
    return not residue


def is_navigation_block(body: str, chinese_context: str, preceding_text: str) -> bool:
    plain_lines = [unquote(line).strip() for line in body.splitlines()]
    nonblank = [line for line in plain_lines if line]
    lower = "\n".join(nonblank).lower()

    title = lower.strip("*_# :.-")
    if len(nonblank) <= 3 and (
        title in NAV_EN_TITLES or title.startswith(NAV_EN)
    ):
        return True

    first_title = nonblank[0].lower().strip("*_# :.-") if nonblank else ""
    first_title = re.sub(r"^\[![^]]+\][+-]?\s*", "", first_title).strip(
        "*_# :.-"
    )
    if first_title in NAV_EN_TITLES or first_title.startswith(NAV_EN):
        return True

    headings = re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", preceding_text)
    current_heading = headings[-1] if headings else ""
    in_navigation_section = any(keyword in current_heading for keyword in NAV_ZH)

    without_title, _ = drop_duplicate_callout_title(body.splitlines())
    structural = [unquote(line).strip() for line in without_title]
    structural = [line for line in structural if line]
    if not structural:
        return False

    # A second-language copy of a pure link list adds no explanatory content.
    # Keep the original list once, regardless of whether its enclosing heading
    # explicitly says "navigation" or "materials".
    if all(is_link_structure(line) for line in structural):
        return True

    if (
        in_navigation_section
        and all(
            bool(re.match(r"^(?:[-+*]\s+|\||\d+[.)]\s)", line))
            for line in structural
        )
        and all(line.startswith("|") or "[[" in line or "](" in line for line in structural)
    ):
        return True

    material_target = bool(
        re.search(
            r"(?:\.pdf|\.pptx?|\.docx?|\.canvas|_Problems|_Solutions|_Lecture_Notes|Exercise\d+)",
            lower,
            re.IGNORECASE,
        )
    )
    if material_target and all(is_link_structure(line) for line in structural):
        return True

    return bool(
        all(line.startswith("|") for line in structural)
        and any("[[" in line for line in structural)
        and any(
            word in lower
            for word in ("official location", "quick jump", "session", "section")
        )
    )


def quote_depth(prefix: str, preceding_line: str) -> int:
    if prefix:
        return prefix.count(">")
    match = re.match(r"^\s*(>+)", preceding_line)
    return len(match.group(1)) if match else 0


def normalize_callout_block(body: str, depth: int) -> tuple[str, bool]:
    lines, dropped_title = drop_duplicate_callout_title(body.splitlines())
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not any(line.strip() for line in lines):
        return "", dropped_title

    quote = ">" * depth
    quoted_lines: list[str] = []
    for line in lines:
        visible = unquote(line).rstrip() if line.strip() else ""
        quoted_lines.append(f"{quote} {visible}" if visible else quote)

    normalized = "\n".join(
        [
            f"{quote} <!-- bilingual-en:start -->",
            *quoted_lines,
            f"{quote} <!-- bilingual-en:end -->",
        ]
    )
    return normalized, dropped_title


def split_callout_prelude(body: str) -> tuple[list[str], str, bool]:
    lines = body.splitlines()
    title_index = None
    for index, line in enumerate(lines):
        visible = unquote(line).strip()
        if visible.startswith("[!") or re.fullmatch(
            r"(?:\*\*|__|\*)(.+?)(?:\*\*|__|\*)", visible
        ):
            title_index = index
            break

    if not title_index:
        return [], body, False

    prelude = lines[:title_index]
    meaningful = [unquote(line).strip() for line in prelude if line.strip()]
    if not meaningful:
        return [], "\n".join(lines[title_index:]), False

    def structural(line: str) -> bool:
        return bool(
            line == "---"
            or line.startswith("#")
            or line.startswith(("- ", "* ", "+ ", "|", "[[", "![["))
            or re.match(r"^\d+[.)]\s", line)
        )

    if not all(structural(line) for line in meaningful):
        return [], body, False

    keep = any(line == "---" or line.startswith("#") for line in meaningful)
    return prelude if keep else [], "\n".join(lines[title_index:]), True


def strip_english(text: str) -> str:
    return BLOCK_RE.sub("", text)


def transform(text: str) -> tuple[str, dict[str, int]]:
    matches = list(BLOCK_RE.finditer(text))
    pieces: list[str] = []
    cursor = 0
    previous_block_end = 0
    counts = {
        "navigation_removed": 0,
        "callouts_normalized": 0,
        "titles_removed": 0,
        "mixed_preludes_removed": 0,
        "heading_preludes_moved": 0,
    }

    for match in matches:
        segment = text[cursor : match.start()]
        chinese_context = text[previous_block_end : match.start()]
        preceding_line = next(
            (line for line in reversed(text[: match.start()].splitlines()) if line.strip()),
            "",
        )

        body = match.group("body")
        lead = match.group("lead")
        prefix = match.group("prefix")

        if is_navigation_block(body, chinese_context, text[: match.start()]):
            replacement = ""
            counts["navigation_removed"] += 1
        else:
            depth = quote_depth(prefix, preceding_line)
            if depth:
                prelude, callout_body, removed_prelude = split_callout_prelude(body)
                if removed_prelude:
                    counts["mixed_preludes_removed"] += 1
                if prelude:
                    callout_start = segment.rfind("> [!")
                    if callout_start < 0:
                        raise RuntimeError("could not locate the Chinese callout header")
                    prepared_prelude: list[str] = []
                    for line in prelude:
                        visible = unquote(line).strip()
                        if not visible or visible == "---":
                            continue
                        heading = re.match(r"^#{1,6}\s+(.+)$", visible)
                        prepared_prelude.append(
                            f"*{heading.group(1)}*" if heading else visible
                        )
                    english_prelude = "\n".join(prepared_prelude)
                    inserted = (
                        "<!-- bilingual-en:start -->\n"
                        f"{english_prelude}\n"
                        "<!-- bilingual-en:end -->"
                    )
                    segment = (
                        segment[:callout_start]
                        + inserted
                        + "\n"
                        + segment[callout_start:]
                    )
                    counts["heading_preludes_moved"] += 1

                normalized, dropped = normalize_callout_block(callout_body, depth)
                replacement = f"{lead}{normalized}" if normalized else ""
                counts["callouts_normalized"] += 1
                counts["titles_removed"] += int(dropped)
            else:
                replacement = match.group(0)

        pieces.append(segment)
        pieces.append(replacement)
        cursor = match.end()
        previous_block_end = match.end()

    pieces.append(text[cursor:])
    result = "".join(pieces)

    before = hashlib.sha256(strip_english(text).encode()).hexdigest()
    after = hashlib.sha256(strip_english(result).encode()).hexdigest()
    if before != after:
        raise RuntimeError(f"source layer changed: {before} != {after}")
    return result, counts


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
        if line.startswith("@@"):
            hunks.append("@@\n")
        else:
            hunks.append(line)

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
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    original = args.path.read_text()
    revised, counts = transform(original)
    if args.summary:
        print(counts)
        return
    print(make_patch(args.path, original, revised), end="")


if __name__ == "__main__":
    main()
