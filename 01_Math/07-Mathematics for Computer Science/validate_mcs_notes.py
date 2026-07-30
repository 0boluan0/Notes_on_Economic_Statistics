#!/usr/bin/env python3
"""Validate the local MIT 6.042J note set without changing the vault."""

from __future__ import annotations

import csv
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:  # The course sync environment already provides PyYAML in this vault.
    yaml = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


COURSE = Path(__file__).resolve().parent
VAULT = COURSE.parents[1]
MATERIALS = COURSE / "MIT_OCW_6.042J_Materials"
FIGURES = VAULT / "98_attachment" / "mathematics_for_computer_science" / "mit6_042j"

NOTES = {
    "01_Proofs.md": range(1, 12),
    "02_Structures.md": range(12, 23),
    "03_Counting.md": range(23, 28),
    "04_Probability.md": range(28, 36),
}
EXPECTED_PSETS = set(range(1, 13))
EXPECTED_EXAMS = {"Midterm 1", "Midterm 2", "Midterm 3", "Final Exam"}
EXPECTED_FIGURES = {
    "unit01-proof-implication.png",
    "unit01-set-operations.png",
    "unit01-induction-dominoes.png",
    "unit01-well-ordering-descent.png",
    "unit01-state-machine-invariant.png",
    "unit01-recursive-structure.png",
    "unit01-cantor-diagonal.png",
    "unit01-proof-method-map.png",
    "unit02-euclidean-algorithm.png",
    "unit02-modular-clock.png",
    "unit02-rsa-flow.png",
    "unit02-directed-walk.png",
    "unit02-dag-topological-order.png",
    "unit02-partial-order-hasse.png",
    "unit02-handshake-lemma.png",
    "unit02-graph-coloring.png",
    "unit02-spanning-tree.png",
    "unit02-stable-matching.png",
    "unit03-sum-product-rule.png",
    "unit03-binomial-paths.png",
    "unit03-stars-and-bars.png",
    "unit03-pigeonhole-principle.png",
    "unit03-inclusion-exclusion.png",
    "unit04-sample-space-events.png",
    "unit04-bayes-tree.png",
    "unit04-independence-grid.png",
    "unit04-random-variable-pmf.png",
    "unit04-expectation-variance.png",
    "unit04-concentration-bounds.png",
    "unit04-random-walk-pagerank.png",
}


def load_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("missing bounded YAML frontmatter")
    raw = text.split("---", 2)[1]
    if yaml is None:
        return {}
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("frontmatter is not a mapping")
    return data


def note_files() -> list[Path]:
    names = ["00_MIT OCW 6.042J course map.md", *NOTES, "05_Review and exam roadmap.md"]
    return [COURSE / name for name in names]


def validate_structure(issues: list[str]) -> None:
    all_sessions: list[int] = []
    psets: set[int] = set()
    exams: set[str] = set()
    for filename, expected in NOTES.items():
        path = COURSE / filename
        if not path.exists():
            issues.append(f"missing main note: {filename}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        sessions = [int(value) for value in re.findall(r"^## Session (\d+)\b", text, re.M)]
        if sessions != list(expected):
            issues.append(f"{filename}: sessions {sessions}, expected {list(expected)}")
        all_sessions.extend(sessions)
        psets.update(int(value) for value in re.findall(r"^## Problem Set (\d+)\b", text, re.M))
        exams.update(re.findall(r"^## (Midterm [123])\b", text, re.M))
    review = COURSE / "05_Review and exam roadmap.md"
    if review.exists() and re.search(r"^##+ .*Final Exam", review.read_text(encoding="utf-8"), re.M):
        exams.add("Final Exam")
    if all_sessions != list(range(1, 36)):
        issues.append(f"course session order is not exactly 1..35: {all_sessions}")
    if psets != EXPECTED_PSETS:
        issues.append(f"problem sets found {sorted(psets)}, expected 1..12")
    if exams != EXPECTED_EXAMS:
        issues.append(f"exams found {sorted(exams)}, expected {sorted(EXPECTED_EXAMS)}")


def build_note_index() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    stems: dict[str, list[Path]] = defaultdict(list)
    aliases: dict[str, list[Path]] = defaultdict(list)
    for path in VAULT.rglob("*.md"):
        if any(part in {".git", ".obsidian"} for part in path.parts):
            continue
        stems[path.stem].append(path)
        if yaml is None:
            continue
        try:
            data = load_frontmatter(path)
        except Exception:
            continue
        for alias in data.get("aliases", []) or []:
            if isinstance(alias, str):
                aliases[alias].append(path)
    return stems, aliases


def resolve_path_target(source: Path, target: str) -> list[Path]:
    candidates = []
    bases = [source.parent / target, VAULT / target, COURSE / target]
    known_suffixes = {".md", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".srt"}
    has_known_suffix = Path(target).suffix.lower() in known_suffixes
    suffixes = [""] if has_known_suffix else ["", ".md", ".pdf", ".png", ".jpg", ".gif", ".srt"]
    for base in bases:
        for suffix in suffixes:
            candidate = Path(str(base) + suffix)
            if candidate.exists() and candidate.is_file():
                candidates.append(candidate.resolve())
    return list(dict.fromkeys(candidates))


def validate_wikilinks(issues: list[str]) -> None:
    stems, aliases = build_note_index()
    pattern = re.compile(r"!?\[\[([^\]]+)\]\]")
    factor_targets: set[Path] = set()
    sources = [*note_files(), MATERIALS / "index.md"]
    for source in sources:
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for raw in pattern.findall(text):
            destination = raw.split("|", 1)[0].split("^", 1)[0].strip()
            target, separator, anchor = destination.partition("#")
            target = target.strip()
            anchor = anchor.strip()
            if not target and separator and anchor:
                matches = [source.resolve()]
            elif not target:
                continue
            else:
                known_suffixes = {".md", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".srt"}
                has_known_suffix = Path(target).suffix.lower() in known_suffixes
                matches = resolve_path_target(source, target) if "/" in target or has_known_suffix else []
                if not matches:
                    stem = Path(target).stem if has_known_suffix else Path(target).name
                    matches = list(dict.fromkeys(stems.get(stem, []) + aliases.get(target, [])))
            if not matches:
                issues.append(f"{source.name}: unresolved wikilink [[{target}]]")
                continue
            if len(matches) > 1:
                choices = ", ".join(str(path.relative_to(VAULT)) for path in matches[:5])
                issues.append(f"{source.name}: ambiguous wikilink [[{target}]] -> {choices}")
                continue
            if anchor and not anchor.startswith("page=") and matches[0].suffix.lower() == ".md":
                destination_text = matches[0].read_text(encoding="utf-8", errors="replace")
                headings = set(re.findall(r"^#{1,6}\s+(.+?)\s*$", destination_text, re.M))
                if anchor not in headings:
                    issues.append(
                        f"{source.name}: unresolved heading [[{target}#{anchor}]]"
                    )
            factor_matches = [path for path in matches if VAULT / "00_factor" in path.parents]
            factor_targets.update(factor_matches)
    for path in sorted(factor_targets):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "course-backlinks-panel" not in text or "```dataview" not in text:
            issues.append(f"factor card lacks course backlink panel: {path.relative_to(VAULT)}")


def validate_markdown(issues: list[str]) -> None:
    bad_latex = re.compile(
        r"(?<!\\)\b(qquad|qquad|quad|binom|frac|ldots|cdots|operatorname|mathbb|infty|boxed|pmod|varphi)\b"
        r"|(?<!\\)\b(left|right)(?=\s*[\(\[\{])"
    )
    for path in note_files():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            load_frontmatter(path)
        except Exception as exc:
            issues.append(f"{path.name}: YAML error: {exc}")
        if text.count("$$") % 2:
            issues.append(f"{path.name}: odd number of $$ delimiters")
        if len(re.findall(r"(?<!\\)\$", text)) % 2:
            issues.append(f"{path.name}: odd number of unescaped $ delimiters")
        if text.count("```") % 2:
            issues.append(f"{path.name}: unclosed fenced code block")
        begin_envs = Counter(re.findall(r"\\begin\{([^}]+)\}", text))
        end_envs = Counter(re.findall(r"\\end\{([^}]+)\}", text))
        if begin_envs != end_envs:
            issues.append(
                f"{path.name}: LaTeX environments differ: begin={dict(begin_envs)}, "
                f"end={dict(end_envs)}"
            )
        brace_stack: list[int] = []
        brace_errors: list[int] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\\" and index + 1 < len(text) and text[index + 1] in "{}":
                index += 2
                continue
            if char == "{":
                brace_stack.append(index)
            elif char == "}":
                if brace_stack:
                    brace_stack.pop()
                else:
                    brace_errors.append(index)
            index += 1
        brace_errors.extend(brace_stack)
        if brace_errors:
            lines = sorted({text.count("\n", 0, offset) + 1 for offset in brace_errors})
            issues.append(f"{path.name}: unbalanced unescaped braces near line(s) {lines[:5]}")
        if "\t" in text:
            issues.append(f"{path.name}: contains tab characters")
        controls = sorted({ord(char) for char in text if ord(char) < 32 and char not in "\n\t\r"})
        if controls:
            issues.append(f"{path.name}: contains ASCII control characters {controls}")
        if re.search(r"\b(TODO|TBD|FIXME)\b|待补|占位", text, re.I):
            issues.append(f"{path.name}: contains placeholder text")
        if ".svg" in text:
            issues.append(f"{path.name}: contains an SVG reference")
        for match in bad_latex.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            issues.append(f"{path.name}:{line}: possible missing backslash before {match.group(1)}")


def validate_pdf_page_links(issues: list[str]) -> None:
    if PdfReader is None:
        issues.append("pypdf is unavailable; PDF page anchors were not checked")
        return
    pattern = re.compile(r"!?\[\[([^\]|#]+\.pdf)#page=(\d+)(?:\|[^\]]*)?\]\]", re.I)
    page_counts: dict[Path, int] = {}
    for source in note_files():
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for target, page_text in pattern.findall(text):
            matches = resolve_path_target(source, target.strip())
            if len(matches) != 1:
                continue  # The general wikilink validator reports this case.
            pdf = matches[0]
            if pdf not in page_counts:
                try:
                    page_counts[pdf] = len(PdfReader(str(pdf)).pages)
                except Exception as exc:
                    issues.append(f"{source.name}: cannot inspect {target}: {exc}")
                    continue
            page = int(page_text)
            if page < 1 or page > page_counts[pdf]:
                issues.append(
                    f"{source.name}: PDF page anchor {target}#page={page} exceeds "
                    f"1..{page_counts[pdf]}"
                )


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def validate_figures(issues: list[str]) -> None:
    actual = {path.name for path in FIGURES.glob("unit*.png")}
    if actual != EXPECTED_FIGURES:
        issues.append(f"figure set differs: missing={sorted(EXPECTED_FIGURES-actual)}, extra={sorted(actual-EXPECTED_FIGURES)}")
    for name in sorted(EXPECTED_FIGURES & actual):
        try:
            size = png_size(FIGURES / name)
        except Exception as exc:
            issues.append(f"{name}: {exc}")
            continue
        if size != (1600, 900):
            issues.append(f"{name}: size {size}, expected (1600, 900)")
    combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in note_files() if path.exists())
    embedded = set(re.findall(r"!\[\[[^\]]*(unit\d\d-[^\]|]+\.png)", combined))
    if embedded != EXPECTED_FIGURES:
        issues.append(f"embedded figure set differs: missing={sorted(EXPECTED_FIGURES-embedded)}, extra={sorted(embedded-EXPECTED_FIGURES)}")


def validate_resource_references(issues: list[str]) -> None:
    """Ensure every teaching asset has a direct entry from the course notes."""

    manifest_path = MATERIALS / "manifest.csv"
    if not manifest_path.exists():
        issues.append("missing manifest.csv")
        return
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in note_files()
        if path.exists()
    )
    target_list = [
        raw.split("|", 1)[0].split("#", 1)[0].split("^", 1)[0].strip()
        for raw in re.findall(r"!?\[\[([^\]]+)\]\]", combined)
    ]
    targets = set(target_list)
    target_counts = Counter(target_list)
    required = {
        "book",
        "session_reading",
        "lecture_slide",
        "video_transcript",
        "in_class_questions",
        "problem_set",
        "exam",
        "courseware_exercise",
    }
    missing: dict[str, list[str]] = defaultdict(list)
    duplicated_exercises: list[str] = []
    for row in manifest:
        category = row["category"]
        if category not in required:
            continue
        local_path = row["local_path"]
        canonical = f"MIT_OCW_6.042J_Materials/{local_path}"
        candidates = {canonical}
        if local_path.endswith(".md"):
            candidates.add(canonical[:-3])
        if not candidates & targets:
            missing[category].append(local_path)
        elif category == "courseware_exercise":
            count = sum(target_counts[candidate] for candidate in candidates)
            if count != 1:
                duplicated_exercises.append(f"{local_path} ({count} links)")
    for category in sorted(missing):
        paths = missing[category]
        issues.append(
            f"main notes miss {len(paths)} direct {category} reference(s): "
            + ", ".join(paths[:3])
        )
    if duplicated_exercises:
        issues.append(
            "courseware exercise links are not one-to-one: "
            + ", ".join(duplicated_exercises[:5])
        )


def validate_courseware_order(issues: list[str]) -> None:
    path = MATERIALS / "courseware_blocks.json"
    if not path.exists():
        issues.append("missing courseware_blocks.json")
        return
    courseware = json.loads(path.read_text(encoding="utf-8"))
    note_for_unit = {
        1: "01_Proofs.md",
        2: "02_Structures.md",
        3: "03_Counting.md",
        4: "04_Probability.md",
    }
    for unit, note_name in note_for_unit.items():
        expected_transcripts: list[str] = []
        expected_exercises: list[str] = []
        for session in courseware["sessions"]:
            if int(session["unit"]) != unit:
                continue
            for block in session["blocks"]:
                if block["type"] == "video":
                    expected_transcripts.append(
                        f"MIT_OCW_6.042J_Materials/{block['transcript_pdf']}"
                    )
                elif block["type"] == "exercise":
                    expected_exercises.append(
                        f"MIT_OCW_6.042J_Materials/{block['exercise_markdown']}"
                    )
        note_text = (COURSE / note_name).read_text(encoding="utf-8", errors="replace")
        raw_targets = [
            raw.split("|", 1)[0].split("#", 1)[0].split("^", 1)[0].strip()
            for raw in re.findall(r"!?\[\[([^\]]+)\]\]", note_text)
        ]
        actual_transcripts = [
            target for target in raw_targets if "/03_Video_Transcripts/" in target
        ]
        actual_exercises = []
        for target in raw_targets:
            if "/08_Courseware_Exercises/" not in target:
                continue
            actual_exercises.append(target if target.endswith(".md") else f"{target}.md")
        if actual_transcripts != expected_transcripts:
            issues.append(
                f"{note_name}: transcript order/count differs "
                f"({len(actual_transcripts)} vs {len(expected_transcripts)})"
            )
        if actual_exercises != expected_exercises:
            issues.append(
                f"{note_name}: exercise block order/count differs "
                f"({len(actual_exercises)} vs {len(expected_exercises)})"
            )


def validate_coverage(issues: list[str]) -> None:
    conflict_copies = sorted(MATERIALS.rglob("* 2.*"))
    if conflict_copies:
        issues.append(f"materials contain {len(conflict_copies)} iCloud conflict-copy files")
    manifest_path = MATERIALS / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        expected_files = {row["local_path"] for row in manifest_rows} | {
            "manifest.csv",
            "index.md",
            "courseware_blocks.json",
            "problem_coverage.csv",
        }
        actual_files = {
            path.relative_to(MATERIALS).as_posix()
            for path in MATERIALS.rglob("*")
            if path.is_file()
        }
        unexpected = sorted(actual_files - expected_files)
        conflict_relatives = {
            path.relative_to(MATERIALS).as_posix() for path in conflict_copies
        }
        untracked_non_conflicts = sorted(set(unexpected) - conflict_relatives)
        if untracked_non_conflicts:
            issues.append(
                f"materials contain {len(untracked_non_conflicts)} other untracked file(s): "
                + ", ".join(untracked_non_conflicts[:3])
            )
    path = MATERIALS / "problem_coverage.csv"
    if not path.exists():
        issues.append("missing problem_coverage.csv")
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    categories = Counter(row["category"] for row in rows)
    expected = Counter({"online_feedback": 376, "problem_set": 38, "in_class": 153, "exam": 29})
    if len(rows) != 596 or categories != expected:
        issues.append(f"coverage ledger has {len(rows)} rows and {dict(categories)}, expected 596 and {dict(expected)}")
    ids = [row["coverage_id"] for row in rows]
    if len(ids) != len(set(ids)):
        issues.append("coverage ledger contains duplicate IDs")

    pending = [row["coverage_id"] for row in rows if "pending" in row["solution_status"].lower()]
    if pending:
        issues.append(
            f"coverage ledger still has {len(pending)} pending rows: {', '.join(pending[:5])}"
        )

    missing_sources = [
        row["coverage_id"] for row in rows if not (MATERIALS / row["source_file"]).is_file()
    ]
    if missing_sources:
        issues.append(
            "coverage ledger has missing source files for "
            + ", ".join(missing_sources[:5])
        )

    headings: dict[str, set[str]] = {}
    for note_name in {row["note_file"] for row in rows}:
        note_path = COURSE / note_name
        if not note_path.is_file():
            headings[note_name] = set()
            continue
        text = note_path.read_text(encoding="utf-8", errors="replace")
        headings[note_name] = set(re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.M))
    bad_anchors = []
    for row in rows:
        anchor = row["note_anchor"]
        if not anchor.startswith("#") or anchor[1:] not in headings.get(row["note_file"], set()):
            bad_anchors.append(row["coverage_id"])
    if bad_anchors:
        issues.append(
            f"coverage ledger has {len(bad_anchors)} non-resolving note anchors: "
            + ", ".join(bad_anchors[:5])
        )


def main() -> int:
    issues: list[str] = []
    validate_structure(issues)
    validate_markdown(issues)
    validate_pdf_page_links(issues)
    validate_wikilinks(issues)
    validate_figures(issues)
    validate_resource_references(issues)
    validate_courseware_order(issues)
    validate_coverage(issues)
    if issues:
        print(f"MIT 6.042J note validation failed with {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("MIT 6.042J notes verified: Sessions 1–35, PS1–12, 4 exams, 596 problems, 30 PNG figures, links and syntax OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
