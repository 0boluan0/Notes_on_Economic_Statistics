#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将课程笔记中的知识点与00_factor卡片建立双链。

能力：
1. full/incremental 两种处理模式
2. dry-run 预览与报告输出
3. 仅处理增量文件（由 changed-files-file 指定）
4. 自动注入 00_factor 课程反链 Dataview 面板（幂等）
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

BACKLINK_PANEL_MARKER = "<!-- course-backlinks-panel -->"
BACKLINK_PANEL_BLOCK = (
    "## 课程笔记反链\n\n"
    f"{BACKLINK_PANEL_MARKER}\n"
    "```dataview\n"
    "LIST FROM \"\"\n"
    "WHERE (\n"
    "  contains(file.path, \"01_Math/\") OR\n"
    "  contains(file.path, \"02_Economy/\") OR\n"
    "  contains(file.path, \"03_Computer_Science/\")\n"
    ") AND contains(file.outlinks, this.file.link)\n"
    "SORT file.mtime DESC\n"
    "```\n"
)


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_frontmatter(content):
    """解析 frontmatter，返回 (frontmatter_dict, body_text, has_frontmatter)。"""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if not match:
        return {}, content, False

    frontmatter_text = match.group(1)
    body = content[match.end():]
    try:
        data = yaml.safe_load(frontmatter_text) or {}
    except Exception:
        data = {}
    return data, body, True


def collect_ranges(pattern, text):
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def in_any_range(start, end, ranges):
    for left, right in ranges:
        if start >= left and end <= right:
            return True
    return False


class KnowledgeBase:
    """00_factor 知识点索引库"""

    def __init__(self, factor_dir):
        self.factor_dir = Path(factor_dir)
        self.entries = []
        self.keyword_map = {}
        self.pattern_map = {}

    @staticmethod
    def _is_alias_usable(alias, filename):
        """
        过滤高误报alias：
        - 保留中文、短语、缩写（含数字或>=2个大写）
        - 过滤“普通英文单词”型alias（如 Multi/Price/Long），除非与文件名同名
        """
        alias = alias.strip()
        if not alias:
            return False

        if alias.lower() == filename.lower():
            return True

        if re.search(r"[\u4e00-\u9fff]", alias):
            return True

        if " " in alias:
            return True

        if any(ch.isdigit() for ch in alias):
            return True

        if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", alias):
            upper_count = sum(ch.isupper() for ch in alias)
            if upper_count >= 2:
                return True
            return False

        return True

    def load_entries(self):
        for md_file in self.factor_dir.rglob("*.md"):
            if md_file.name.startswith(".") or md_file.name.startswith("_"):
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as exc:
                print(f"警告：无法读取卡片 {md_file}: {exc}")
                continue

            frontmatter, _, _ = parse_frontmatter(content)
            filename = md_file.stem
            aliases = frontmatter.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [x for x in aliases if isinstance(x, str)]

            entry = {
                "filename": filename,
                "path": str(md_file),
                "aliases": aliases,
                "subject": frontmatter.get("科目", ""),
                "tags": frontmatter.get("tags", []),
            }
            self.entries.append(entry)

            keywords = [filename]
            for alias in aliases:
                if self._is_alias_usable(alias, filename):
                    keywords.append(alias)

            for keyword in keywords:
                normalized = keyword.strip().lower()
                if not normalized:
                    continue
                bucket = self.keyword_map.setdefault(normalized, [])
                if filename not in [e["filename"] for e in bucket]:
                    bucket.append(entry)

        print(f"成功加载 {len(self.entries)} 个知识点卡片")
        print(f"建立了 {len(self.keyword_map)} 个关键词映射")

        # 预编译关键词正则，避免每个文件重复编译导致性能下降
        for keyword in self.keyword_map.keys():
            if re.match(r"^[a-zA-Z0-9_]+$", keyword):
                pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
            else:
                pattern = re.compile(
                    r"(?<![\w\u4e00-\u9fff])" + re.escape(keyword) + r"(?![\w\u4e00-\u9fff])",
                    re.IGNORECASE,
                )
            self.pattern_map[keyword] = pattern

    def find_matches(self, text):
        """
        在正文中查找候选匹配。
        返回: (matches, stats)
        """
        stats = {
            "skipped_special_blocks": 0,
            "skipped_existing_links": 0,
            "skipped_overlap": 0,
        }

        matches = []
        sorted_keywords = sorted(self.keyword_map.keys(), key=len, reverse=True)
        text_lower = text.lower()

        code_ranges = collect_ranges(re.compile(r"```[\s\S]*?```"), text)
        latex_block_ranges = collect_ranges(re.compile(r"\$\$[\s\S]*?\$\$"), text)
        latex_inline_ranges = collect_ranges(re.compile(r"\$[^\$]*?\$"), text)
        link_ranges = collect_ranges(re.compile(r"\[\[[^\[\]]+?\]\]"), text)

        special_ranges = code_ranges + latex_block_ranges + latex_inline_ranges

        for keyword in sorted_keywords:
            if len(keyword) < 3:
                continue
            if re.search(r"[\u4e00-\u9fff]", keyword) and len(keyword) < 4:
                continue

            # 先做低成本子串过滤，减少正则匹配次数
            if keyword not in text_lower:
                continue

            pattern = self.pattern_map[keyword]

            for match in pattern.finditer(text):
                start, end = match.start(), match.end()

                if in_any_range(start, end, special_ranges):
                    stats["skipped_special_blocks"] += 1
                    continue

                if in_any_range(start, end, link_ranges):
                    stats["skipped_existing_links"] += 1
                    continue

                matches.append(
                    {
                        "keyword": keyword,
                        "start": start,
                        "end": end,
                        "entries": self.keyword_map[keyword],
                    }
                )

        matches.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
        filtered = []
        for candidate in matches:
            overlap = False
            for existing in filtered:
                if candidate["start"] < existing["end"] and candidate["end"] > existing["start"]:
                    overlap = True
                    stats["skipped_overlap"] += 1
                    break
            if not overlap:
                filtered.append(candidate)

        filtered.sort(key=lambda x: x["start"])
        return filtered, stats


class LinkGenerator:
    """课程笔记 -> 00_factor 链接生成"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def _resolve_entry(self, entries, keyword_text):
        if len(entries) == 1:
            return entries[0], None

        text_l = keyword_text.lower()
        exact = []
        for entry in entries:
            aliases_l = [a.lower() for a in entry.get("aliases", []) if isinstance(a, str)]
            if text_l == entry["filename"].lower() or text_l in aliases_l:
                exact.append(entry)

        if len(exact) == 1:
            return exact[0], None

        options = sorted({e["filename"] for e in entries})
        return None, options

    def process_content(self, content, file_path):
        """返回 (new_content, note_stats)。"""
        _, body, has_frontmatter = parse_frontmatter(content)
        prefix = content[: len(content) - len(body)] if has_frontmatter else ""

        matches, match_stats = self.kb.find_matches(body)

        stats = {
            "file": str(file_path),
            "matched_candidates": len(matches),
            "links_added": 0,
            "ambiguous_skipped": 0,
            "ambiguous_samples": [],
            "skipped_special_blocks": match_stats["skipped_special_blocks"],
            "skipped_existing_links": match_stats["skipped_existing_links"],
            "skipped_overlap": match_stats["skipped_overlap"],
        }

        modified_body = body
        for match in reversed(matches):
            start, end = match["start"], match["end"]
            raw_text = body[start:end]

            entry, options = self._resolve_entry(match["entries"], raw_text)
            if entry is None:
                stats["ambiguous_skipped"] += 1
                if len(stats["ambiguous_samples"]) < 20:
                    stats["ambiguous_samples"].append(
                        {
                            "keyword": raw_text,
                            "options": options,
                            "position": [start, end],
                        }
                    )
                continue

            link = f"[[{entry['filename']}|{raw_text}]]"
            modified_body = modified_body[:start] + link + modified_body[end:]
            stats["links_added"] += 1

        return prefix + modified_body, stats


class Linker:
    """执行器"""

    def __init__(
        self,
        source_dirs,
        factor_dir,
        mode,
        changed_files,
        output_dir=None,
        dry_run=False,
        inject_backlink_panel=True,
    ):
        self.source_dirs = [Path(p).resolve() for p in source_dirs]
        self.factor_dir = Path(factor_dir).resolve()
        self.mode = mode
        self.changed_files = [Path(p).resolve() for p in changed_files]
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.dry_run = dry_run
        self.inject_backlink_panel = inject_backlink_panel

        self.kb = KnowledgeBase(self.factor_dir)
        self.kb.load_entries()
        self.link_generator = LinkGenerator(self.kb)

    def _resolve_output_path(self, source_file):
        if not self.output_dir:
            return source_file

        for source_dir in self.source_dirs:
            try:
                rel = source_file.relative_to(source_dir)
                output_path = self.output_dir / source_dir.name / rel
                output_path.parent.mkdir(parents=True, exist_ok=True)
                return output_path
            except ValueError:
                continue

        output_path = self.output_dir / source_file.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _collect_full_files(self):
        files = []
        for source_dir in self.source_dirs:
            if not source_dir.exists() or not source_dir.is_dir():
                print(f"警告：目录 {source_dir} 不存在或不是目录")
                continue

            for md_file in source_dir.rglob("*.md"):
                if md_file.name.startswith("."):
                    continue
                files.append(md_file.resolve())

        return sorted(set(files))

    def _collect_incremental_files(self):
        files = []
        source_dir_set = set(self.source_dirs)

        for file_path in self.changed_files:
            if file_path.suffix.lower() != ".md":
                continue
            if not file_path.exists() or not file_path.is_file():
                continue

            for source_dir in source_dir_set:
                try:
                    file_path.relative_to(source_dir)
                    files.append(file_path)
                    break
                except ValueError:
                    continue

        return sorted(set(files))

    def _build_backlink_panel(self, content):
        if BACKLINK_PANEL_MARKER in content:
            return content, False

        new_content = content
        if not new_content.endswith("\n"):
            new_content += "\n"
        if not new_content.endswith("\n\n"):
            new_content += "\n"
        new_content += BACKLINK_PANEL_BLOCK
        return new_content, True

    def _inject_backlink_panels(self):
        stats = {
            "factor_total_files": 0,
            "backlink_panel_candidates": 0,
            "backlink_panel_inserted": 0,
            "errors": [],
        }

        if not self.inject_backlink_panel:
            return stats

        for md_file in sorted(self.factor_dir.rglob("*.md")):
            if md_file.name.startswith("."):
                continue

            stats["factor_total_files"] += 1

            try:
                content = md_file.read_text(encoding="utf-8")
                updated, changed = self._build_backlink_panel(content)
                if not changed:
                    continue

                stats["backlink_panel_candidates"] += 1
                if not self.dry_run:
                    md_file.write_text(updated, encoding="utf-8")
                stats["backlink_panel_inserted"] += 1
            except Exception as exc:
                stats["errors"].append({"file": str(md_file), "error": str(exc)})

        return stats

    def process(self):
        summary = {
            "run_at": now_iso(),
            "mode": self.mode,
            "dry_run": self.dry_run,
            "factor_dir": str(self.factor_dir),
            "source_dirs": [str(p) for p in self.source_dirs],
            "source_files_total": 0,
            "source_files_processed": 0,
            "source_files_changed": 0,
            "links_added": 0,
            "matched_candidates": 0,
            "skipped_special_blocks": 0,
            "skipped_existing_links": 0,
            "skipped_overlap": 0,
            "ambiguous_skipped": 0,
            "ambiguous_samples": [],
            "changed_files": [],
            "errors": [],
            "backlink_panel": {},
        }

        if self.mode == "full":
            targets = self._collect_full_files()
        else:
            targets = self._collect_incremental_files()

        summary["source_files_total"] = len(targets)

        for md_file in targets:
            try:
                original = md_file.read_text(encoding="utf-8")
                updated, note_stats = self.link_generator.process_content(original, md_file)

                summary["source_files_processed"] += 1
                summary["matched_candidates"] += note_stats["matched_candidates"]
                summary["links_added"] += note_stats["links_added"]
                summary["skipped_special_blocks"] += note_stats["skipped_special_blocks"]
                summary["skipped_existing_links"] += note_stats["skipped_existing_links"]
                summary["skipped_overlap"] += note_stats["skipped_overlap"]
                summary["ambiguous_skipped"] += note_stats["ambiguous_skipped"]

                if note_stats["ambiguous_samples"] and len(summary["ambiguous_samples"]) < 50:
                    summary["ambiguous_samples"].append(
                        {
                            "file": str(md_file),
                            "samples": note_stats["ambiguous_samples"],
                        }
                    )

                if updated != original:
                    output_path = self._resolve_output_path(md_file)
                    if not self.dry_run:
                        output_path.write_text(updated, encoding="utf-8")
                    summary["source_files_changed"] += 1
                    if len(summary["changed_files"]) < 500:
                        summary["changed_files"].append(str(output_path))
            except Exception as exc:
                summary["errors"].append({"file": str(md_file), "error": str(exc)})

        summary["backlink_panel"] = self._inject_backlink_panels()
        return summary


def read_changed_files(changed_files_file):
    lines = Path(changed_files_file).read_text(encoding="utf-8").splitlines()
    resolved = []
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        resolved.append(path)
    return resolved


def write_report(report_path, summary):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="将课程笔记中知识点卡片链接化（支持全量/增量）")

    parser.add_argument(
        "--factor-dir",
        default="/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/00_factor",
        help="知识点卡片目录 (00_factor)",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=[
            "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/01_Math",
            "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/02_Economy",
            "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic/03_Computer_Science",
        ],
        help="要处理的笔记目录列表",
    )
    parser.add_argument("--mode", choices=["full", "incremental"], default="full", help="运行模式")
    parser.add_argument("--changed-files-file", help="增量模式下的变更文件列表（每行一个路径）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入文件")
    parser.add_argument("--report-path", help="JSON 报告输出路径")
    parser.add_argument("--output-dir", help="输出目录（不指定则覆盖源文件）")

    parser.add_argument(
        "--inject-backlink-panel",
        dest="inject_backlink_panel",
        action="store_true",
        help="为 00_factor 注入课程反链 Dataview 面板（默认开启）",
    )
    parser.add_argument(
        "--no-inject-backlink-panel",
        dest="inject_backlink_panel",
        action="store_false",
        help="关闭 00_factor 课程反链面板注入",
    )
    parser.set_defaults(inject_backlink_panel=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    changed_files = []
    if args.mode == "incremental":
        if not args.changed_files_file:
            parser.error("incremental 模式需要 --changed-files-file")
        changed_files = read_changed_files(args.changed_files_file)

    linker = Linker(
        source_dirs=args.source_dirs,
        factor_dir=args.factor_dir,
        mode=args.mode,
        changed_files=changed_files,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        inject_backlink_panel=args.inject_backlink_panel,
    )

    summary = linker.process()

    print("\n处理完成")
    print(f"mode: {summary['mode']}")
    print(f"dry_run: {summary['dry_run']}")
    print(f"source_files_total: {summary['source_files_total']}")
    print(f"source_files_processed: {summary['source_files_processed']}")
    print(f"source_files_changed: {summary['source_files_changed']}")
    print(f"links_added: {summary['links_added']}")
    print(f"skipped_special_blocks: {summary['skipped_special_blocks']}")
    print(f"skipped_existing_links: {summary['skipped_existing_links']}")
    print(f"ambiguous_skipped: {summary['ambiguous_skipped']}")
    print(f"backlink_panel_inserted: {summary['backlink_panel'].get('backlink_panel_inserted', 0)}")
    print(f"errors: {len(summary['errors']) + len(summary['backlink_panel'].get('errors', []))}")

    if args.report_path:
        write_report(args.report_path, summary)
        print(f"report: {args.report_path}")


if __name__ == "__main__":
    main()
