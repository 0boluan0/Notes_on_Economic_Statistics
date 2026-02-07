#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import fnmatch
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
FACTOR_DIR = ROOT / "00_factor"
SOURCE_DIRS = [ROOT / "01_Math", ROOT / "02_Economy"]
DEFAULT_BLACKLIST = ROOT / "config" / "link_blacklist.txt"
DEFAULT_WHITELIST = ROOT / "config" / "link_whitelist.txt"

WIKILINK_PATTERN = re.compile(r"!?\[\[[^\]]*\]\]")
CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]*`")

CONFLICT_RANK = {
    "concept": 0,
    "framework": 1,
    "procedure": 2,
    "system": 3,
    "proof": 4,
    "writing": 5,
    "00_hub": 6,
}


@dataclass(frozen=True)
class Candidate:
    card_name: str
    card_type: str
    rel_path: str


@dataclass(frozen=True)
class MatchItem:
    start: int
    end: int
    term: str
    card_name: str


def load_terms_file(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def extract_frontmatter_aliases(text: str) -> List[str]:
    aliases: List[str] = []
    if not text.startswith("---\n"):
        return aliases
    end = text.find("\n---", 4)
    if end == -1:
        return aliases

    fm = text[4:end].splitlines()
    in_aliases = False
    for line in fm:
        stripped = line.strip()
        if stripped.startswith("aliases:"):
            in_aliases = True
            continue
        if in_aliases:
            if line.startswith("- "):
                aliases.append(line[2:].strip().strip('"\''))
            elif line.startswith("  - "):
                aliases.append(line[4:].strip().strip('"\''))
            elif not stripped:
                continue
            else:
                in_aliases = False
    return [a for a in aliases if a]


def term_has_cjk(term: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in term)


def is_ascii_text(s: str) -> bool:
    return all(ord(ch) < 128 for ch in s)


def quality_filter(term: str, level: str, whitelist: Set[str], blacklist: Set[str]) -> bool:
    t = term.strip()
    if len(t) < 2:
        return False
    if t in blacklist:
        return False
    if term_has_cjk(t):
        return True
    if not is_ascii_text(t):
        return False
    if t in whitelist:
        return True

    if level == "strict":
        if t.isupper() and 2 <= len(t) <= 10 and re.fullmatch(r"[A-Z0-9\-]+", t):
            return True
        if " " in t or "-" in t:
            return len(t) >= 4
        return False

    if level == "medium":
        if t.isupper() and 2 <= len(t) <= 10 and re.fullmatch(r"[A-Z0-9\-]+", t):
            return True
        if " " in t or "-" in t:
            return len(t) >= 4
        # medium: allow single English words if title-cased and length >= 5
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]+", t) and len(t) >= 5:
            return True
        return False

    # loose
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9\- ]+", t) and len(t) >= 3:
        return True
    return False


def detect_card_type(md: Path) -> str:
    rel = md.relative_to(FACTOR_DIR).as_posix().split("/")
    return rel[0] if rel else "unknown"


def choose_candidate(cands: Sequence[Candidate], conflict_policy: str) -> Tuple[Candidate | None, str]:
    if not cands:
        return None, "no-candidate"
    if len(cands) == 1:
        return cands[0], "single-candidate"

    if conflict_policy == "skip":
        return None, "conflict-skip"

    if conflict_policy == "prefer-hub":
        hubs = sorted([c for c in cands if c.card_type == "00_hub"], key=lambda c: c.card_name)
        if hubs:
            return hubs[0], "conflict-prefer-hub"

    # default & prefer-concept fallback: deterministic rank
    ranked = sorted(cands, key=lambda c: (CONFLICT_RANK.get(c.card_type, 99), c.card_name.lower()))
    return ranked[0], "conflict-prefer-rank"


def build_term_index(
    matching_level: str,
    conflict_policy: str,
    whitelist: Set[str],
    blacklist: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[Candidate]], Dict[str, str], int]:
    term_to_candidates: Dict[str, List[Candidate]] = defaultdict(list)
    card_count = 0

    for md in FACTOR_DIR.rglob("*.md"):
        card_count += 1
        card_name = md.stem
        card_type = detect_card_type(md)
        rel_path = md.relative_to(ROOT).as_posix()
        candidate = Candidate(card_name=card_name, card_type=card_type, rel_path=rel_path)

        try:
            content = md.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = md.read_text(encoding="utf-8", errors="ignore")

        terms = {card_name}
        terms.update(extract_frontmatter_aliases(content))
        for term in terms:
            t = term.strip()
            if t and quality_filter(t, matching_level, whitelist, blacklist):
                term_to_candidates[t].append(candidate)

    resolved: Dict[str, str] = {}
    chosen_reason: Dict[str, str] = {}
    conflicts: Dict[str, List[Candidate]] = {}
    for term, cands in term_to_candidates.items():
        selected, reason = choose_candidate(cands, conflict_policy)
        if selected:
            resolved[term] = selected.card_name
            chosen_reason[term] = reason
        if len(cands) > 1:
            conflicts[term] = sorted(cands, key=lambda c: c.card_name.lower())

    return resolved, conflicts, chosen_reason, card_count


def iter_source_files(include_globs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for d in SOURCE_DIRS:
        files.extend(sorted(d.rglob("*.md")))

    if not include_globs:
        return files

    selected: List[Path] = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if any(fnmatch.fnmatch(rel, g) for g in include_globs):
            selected.append(p)
    return selected


def collect_protected_ranges(text: str) -> List[Tuple[int, int, str]]:
    ranges: List[Tuple[int, int, str]] = []
    for m in CODE_FENCE_PATTERN.finditer(text):
        ranges.append((m.start(), m.end(), "code_fence"))
    for m in INLINE_CODE_PATTERN.finditer(text):
        ranges.append((m.start(), m.end(), "inline_code"))
    for m in WIKILINK_PATTERN.finditer(text):
        ranges.append((m.start(), m.end(), "embed" if text[m.start()] == "!" else "wikilink"))
    ranges.sort(key=lambda x: x[0])
    return ranges


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def find_overlapping_kind(start: int, end: int, ranges: List[Tuple[int, int, str]]) -> str | None:
    for rs, re_, kind in ranges:
        if overlaps(start, end, rs, re_):
            return kind
    return None


def boundary_ok(text: str, start: int, end: int) -> bool:
    prev_ch = text[start - 1] if start > 0 else ""
    next_ch = text[end] if end < len(text) else ""
    if prev_ch and (prev_ch.isalnum() or prev_ch == "_"):
        return False
    if next_ch and (next_ch.isalnum() or next_ch == "_"):
        return False
    return True


def compute_matches(text: str, sorted_terms: Sequence[str], term_to_card: Dict[str, str], counters: Counter) -> List[MatchItem]:
    protected = collect_protected_ranges(text)
    matches: List[MatchItem] = []
    occupied: List[Tuple[int, int]] = []

    for term in sorted_terms:
        idx = 0
        while True:
            pos = text.find(term, idx)
            if pos == -1:
                break
            start, end = pos, pos + len(term)
            idx = pos + 1

            if not boundary_ok(text, start, end):
                continue

            kind = find_overlapping_kind(start, end, protected)
            if kind:
                counters[f"skip_{kind}"] += 1
                continue

            if any(overlaps(start, end, s, e) for s, e in occupied):
                counters["skip_overlap"] += 1
                continue

            matches.append(MatchItem(start=start, end=end, term=term, card_name=term_to_card[term]))
            occupied.append((start, end))

    matches.sort(key=lambda m: m.start)
    return matches


def apply_matches(text: str, matches: Sequence[MatchItem]) -> Tuple[str, List[Tuple[str, str]]]:
    if not matches:
        return text, []
    out = text
    replacements: List[Tuple[str, str]] = []
    for m in reversed(matches):
        original = out[m.start : m.end]
        linked = f"[[{m.card_name}|{original}]]"
        out = out[: m.start] + linked + out[m.end :]
        replacements.append((original, m.card_name))
    replacements.reverse()
    return out, replacements


def run(args: argparse.Namespace) -> int:
    whitelist = load_terms_file(DEFAULT_WHITELIST)
    blacklist = load_terms_file(Path(args.blacklist) if args.blacklist else DEFAULT_BLACKLIST)

    term_to_card, conflicts, chosen_reason, card_count = build_term_index(
        matching_level=args.matching_level,
        conflict_policy=args.conflict_policy,
        whitelist=whitelist,
        blacklist=blacklist,
    )
    sorted_terms = sorted(term_to_card.keys(), key=lambda x: len(x), reverse=True)
    source_files = iter_source_files(args.include_path)

    stats = Counter()
    per_file: Dict[str, List[Tuple[str, str]]] = {}
    term_hits = Counter()
    write_failed: List[str] = []

    for path in source_files:
        stats["files_scanned"] += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = compute_matches(text, sorted_terms, term_to_card, stats)
        stats["match_hits"] += len(matches)
        for m in matches:
            term_hits[m.term] += 1

        new_text, replacements = apply_matches(text, matches)
        if replacements:
            rel = path.relative_to(ROOT).as_posix()
            per_file[rel] = replacements
            stats["files_with_replacements"] += 1
            stats["actual_replacements"] += len(replacements)
            if args.mode == "apply" and new_text != text:
                try:
                    path.write_text(new_text, encoding="utf-8")
                except PermissionError:
                    stats["skip_write_permission"] += 1
                    write_failed.append(rel)
                except OSError:
                    stats["skip_write_oserror"] += 1
                    write_failed.append(rel)

    print(f"=== link_factor_cards {args.mode.upper()} REPORT ===")
    print(f"cards_indexed: {card_count}")
    print(f"unique_terms: {len(term_to_card)}")
    print(f"conflict_terms: {len(conflicts)}")
    print(f"files_scanned: {stats['files_scanned']}")
    print(f"match_hits: {stats['match_hits']}")
    print(f"actual_replacements: {stats['actual_replacements']}")
    print(f"files_with_replacements: {stats['files_with_replacements']}")
    print("skip_reasons:")
    for key in sorted(k for k in stats if k.startswith("skip_")):
        print(f"  {key}: {stats[key]}")

    print("\nTop matched terms:")
    for term, cnt in term_hits.most_common(20):
        print(f"  {term}: {cnt}")

    print("\nTop files by replacements:")
    top_files = sorted(per_file.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]
    for rel, repl in top_files:
        print(f"  {rel}: {len(repl)}")

    if write_failed:
        print("\nwrite_failed:")
        for rel in write_failed:
            print(f"  {rel}")

    if conflicts:
        print("\nconflict_terms_detail:")
        for term in sorted(conflicts.keys()):
            cards = ", ".join([f"{c.card_name}({c.card_type})" for c in conflicts[term]])
            print(f"  {term} -> {cards}; selected={term_to_card.get(term, 'SKIPPED')} ({chosen_reason.get(term, 'n/a')})")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "matching_level": args.matching_level,
            "conflict_policy": args.conflict_policy,
            "cards_indexed": card_count,
            "unique_terms": len(term_to_card),
            "conflict_terms": len(conflicts),
            "stats": dict(stats),
            "top_terms": term_hits.most_common(100),
            "top_files": [[rel, len(repl)] for rel, repl in sorted(per_file.items(), key=lambda kv: len(kv[1]), reverse=True)],
            "per_file_replacements": {rel: [{"source": s, "target": t} for s, t in repl] for rel, repl in per_file.items()},
            "conflicts": {
                term: {
                    "candidates": [{"card_name": c.card_name, "card_type": c.card_type, "path": c.rel_path} for c in cands],
                    "selected": term_to_card.get(term),
                    "reason": chosen_reason.get(term),
                }
                for term, cands in conflicts.items()
            },
            "write_failed": write_failed,
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        printable = report_path.resolve().relative_to(ROOT).as_posix() if report_path.is_absolute() else report_path.as_posix()
        print(f"\nreport_json: {printable}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link class-note mentions to 00_factor cards.")
    parser.add_argument("--mode", choices=["dry-run", "apply"], help="Execution mode.")
    parser.add_argument("--dry-run", action="store_true", help="Compatibility flag for dry-run.")
    parser.add_argument("--apply", action="store_true", help="Compatibility flag for apply.")
    parser.add_argument("--conflict-policy", choices=["skip", "prefer-concept", "prefer-hub"], default="prefer-concept")
    parser.add_argument("--matching-level", choices=["strict", "medium", "loose"], default="medium")
    parser.add_argument("--include-path", action="append", default=[], help="Glob (relative to repo root), repeatable.")
    parser.add_argument("--report-json", help="Write machine-readable report JSON.")
    parser.add_argument("--blacklist", help="Blacklist file path (one term per line).")
    args = parser.parse_args()

    flags = [args.dry_run, args.apply, args.mode is not None]
    if sum(bool(x) for x in flags) != 1:
        parser.error("Provide exactly one of --mode, --dry-run, --apply")

    if args.mode is None:
        args.mode = "apply" if args.apply else "dry-run"
    return args


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
