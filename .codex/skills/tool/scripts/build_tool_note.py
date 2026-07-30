#!/usr/bin/env python3
"""
Build or update a tool note in 05_tools from a project URL.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


USER_AGENT = "tool-skill-note-builder/1.0"
GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

INSTALL_KEYWORDS = [
    "install",
    "installation",
    "setup",
    "prerequisite",
    "requirements",
    "dependency",
    "安装",
    "部署",
    "环境",
    "依赖",
]
FIRST_RUN_KEYWORDS = [
    "quick start",
    "quickstart",
    "getting started",
    "run",
    "start",
    "demo",
    "首次",
    "快速开始",
    "启动",
    "运行",
]
USAGE_KEYWORDS = [
    "usage",
    "examples",
    "example",
    "guide",
    "tutorial",
    "workflow",
    "cli",
    "command",
    "使用",
    "示例",
    "教程",
    "工作流",
]
TROUBLESHOOT_KEYWORDS = [
    "troubleshoot",
    "troubleshooting",
    "faq",
    "common issues",
    "error",
    "known issues",
    "limitations",
    "排错",
    "问题",
    "常见",
    "注意事项",
]

ILLEGAL_FILENAME_CHARS = r'[\\/:*?"<>|]'
INDEX_FILENAME = "00_content.md"


class ToolNoteError(Exception):
    """Raised when note generation cannot continue safely."""


@dataclass
class RepoInfo:
    owner: str
    repo: str
    html_url: str
    description: str
    homepage: str
    language: str
    stars: Optional[int]
    license_name: str
    topics: List[str]
    updated_at: str


@dataclass
class ReleaseInfo:
    name: str
    tag_name: str
    html_url: str
    published_at: str


@dataclass
class ReadmeInfo:
    text: str
    html_url: str
    download_url: str
    h1: str


@dataclass
class WebsiteInfo:
    url: str
    site_name: str
    title: str
    description: str
    github_url: str


@dataclass
class SectionMatch:
    titles: List[str]
    snippet_lines: List[str]
    code_blocks: List[Tuple[str, str]]


class MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: List[str] = []
        self.in_title = False
        self.meta: Dict[str, str] = {}
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k.lower(): (v or "").strip() for k, v in attrs}
        if tag.lower() == "title":
            self.in_title = True
            return
        if tag.lower() == "a":
            href = attr_map.get("href", "")
            if href:
                self.links.append(href)
            return
        if tag.lower() == "meta":
            key = (attr_map.get("property") or attr_map.get("name") or "").lower()
            content = attr_map.get("content", "")
            if key and content and key not in self.meta:
                self.meta[key] = content

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title and data.strip():
            self.title_parts.append(data.strip())

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()


def fetch_bytes(url: str, extra_headers: Optional[Dict[str, str]] = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            return response.read()
    except (HTTPError, URLError) as exc:
        raise ToolNoteError(f"Failed to fetch URL: {url} ({exc})") from exc


def fetch_text(url: str, extra_headers: Optional[Dict[str, str]] = None) -> str:
    raw = fetch_bytes(url, extra_headers=extra_headers)
    return raw.decode("utf-8", errors="replace")


def fetch_json(url: str, extra_headers: Optional[Dict[str, str]] = None) -> dict:
    text = fetch_text(url, extra_headers=extra_headers)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolNoteError(f"Invalid JSON from {url}") from exc
    if not isinstance(data, dict) and not isinstance(data, list):
        raise ToolNoteError(f"Unexpected JSON payload from {url}")
    return data


def parse_date(value: str) -> str:
    value = value.strip()
    if not value:
        return "unknown"
    try:
        if "T" in value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d")
        parsed = datetime.strptime(value, "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        return value


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(ILLEGAL_FILENAME_CHARS, " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned or "tool-note"


def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = re.sub(r"`{1,3}", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip(" -|>")


def parse_markdown_sections(markdown: str) -> List[Tuple[str, str]]:
    lines = markdown.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = "README"
    current_lines: List[str] = []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            sections.append((current_title, current_lines))
            current_title = normalize_line(match.group(2))
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_title, current_lines))

    output: List[Tuple[str, str]] = []
    for title, content_lines in sections:
        content = "\n".join(content_lines).strip()
        if title and content:
            output.append((title, content))
    return output


def extract_first_h1(markdown: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line.strip())
        if match:
            return normalize_line(match.group(1))
    return ""


def strip_frontmatter(markdown: str) -> str:
    if not markdown.startswith("---\n"):
        return markdown
    parts = markdown.split("\n---\n", 1)
    if len(parts) != 2:
        return markdown
    return parts[1]


def extract_intro_paragraph(markdown: str) -> str:
    body = strip_frontmatter(markdown)
    h1 = re.search(r"^#\s+.+?\s*$", body, re.MULTILINE)
    if not h1:
        return "暂无一句话总结。"
    after_h1 = body[h1.end() :]
    next_section = re.search(r"^##\s+", after_h1, re.MULTILINE)
    if next_section:
        after_h1 = after_h1[: next_section.start()]

    lines: List[str] = []
    for line in after_h1.splitlines():
        stripped = normalize_line(line)
        if not stripped:
            if lines:
                break
            continue
        lines.append(stripped)
    return " ".join(lines) if lines else "暂无一句话总结。"


def table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def extract_section(markdown: str, heading: str) -> str:
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(markdown)
    if not match:
        return ""
    section = markdown[match.end() :]
    next_heading = re.search(r"^##\s+", section, re.MULTILINE)
    return section[: next_heading.start()] if next_heading else section


def extract_bullet_value(markdown: str, label: str) -> str:
    pattern = re.compile(rf"^-\s+{re.escape(label)}：(.+)$", re.MULTILINE)
    match = pattern.search(markdown)
    if not match:
        return ""
    return normalize_line(match.group(1))


def section_state(markdown: str, heading: str) -> str:
    section = extract_section(markdown, heading)
    if not section:
        return "缺章节"
    if "信息不足" in section or "TODO" in section:
        return "待补"
    return "已记录"


def rebuild_directory_page(folder: Path) -> None:
    rows: List[Tuple[str, str, str, str, str, str, str, str]] = []
    for candidate in sorted(folder.glob("*.md")):
        if candidate.name in {INDEX_FILENAME, "目录.md"}:
            continue
        text = candidate.read_text(encoding="utf-8")
        body = strip_frontmatter(text)
        title = extract_first_h1(body) or candidate.stem
        summary = extract_intro_paragraph(text)
        language = extract_bullet_value(body, "主要语言") or extract_bullet_value(body, "技术栈") or "未标注"
        source = extract_bullet_value(body, "GitHub 仓库") or extract_bullet_value(body, "项目地址") or extract_bullet_value(body, "输入链接") or "未标注"
        install = section_state(body, "安装方法（Installation）")
        first_run = section_state(body, "首次使用（First run）")
        daily = section_state(body, "后续使用（Daily usage）")
        missing = ", ".join(name for name, state in [
            ("安装", install),
            ("首次使用", first_run),
            ("后续使用", daily),
        ] if state != "已记录") or "完整"
        link = f"[[{candidate.stem}|{title}]]" if candidate.stem != title else f"[[{title}]]"
        rows.append((title.casefold(), link, language, source, install, first_run, missing, summary))

    needs_cleanup = [row for row in rows if row[6] != "完整"]

    lines = [
        "---",
        "aliases:",
        "  - Tools 目录",
        "tags: [index, tools]",
        "created_by: \"tool\"",
        "---",
        "",
        "# Tools",
        "",
        "> [!summary] 导航",
        f"> 共 {len(rows)} 个工具。摘要取自每篇工具笔记标题后的首段。",
        "",
        "> [!info] 字段说明",
        "> `待补` 表示对应章节仍含 `信息不足` 或 `TODO`。",
        "",
    ]
    if needs_cleanup:
        lines.append("> [!warning] 待补信息")
        for _, link, _, _, _, _, missing, _ in sorted(needs_cleanup):
            lines.append(f"> - {link}：{missing}")
        lines.append("")

    lines.extend([
        "| 工具 | 语言/平台 | 来源 | 安装 | 首次使用 | 待补 | 一句话总结 |",
        "|---|---|---|---|---|---|---|",
    ])
    if rows:
        for _, link, language, source, install, first_run, missing, summary in sorted(rows):
            lines.append(f"| {table_cell(link)} | {table_cell(language)} | {table_cell(source)} | {table_cell(install)} | {table_cell(first_run)} | {table_cell(missing)} | {table_cell(summary)} |")
    else:
        lines.append("- 暂无 tool 笔记。")
    lines.append("")
    write_markdown(folder / INDEX_FILENAME, "\n".join(lines), overwrite=True)


def extract_code_blocks(text: str, max_blocks: int = 2) -> List[Tuple[str, str]]:
    pattern = re.compile(r"```([a-zA-Z0-9_-]*)\n(.*?)```", re.DOTALL)
    output: List[Tuple[str, str]] = []
    for match in pattern.finditer(text):
        lang = (match.group(1) or "").strip() or "bash"
        code = match.group(2).strip()
        if not code:
            continue
        output.append((lang, code))
        if len(output) >= max_blocks:
            break
    return output


def match_sections(readme_text: str, keywords: List[str]) -> SectionMatch:
    sections = parse_markdown_sections(readme_text)
    keys = [k.lower() for k in keywords]
    titles: List[str] = []
    snippets: List[str] = []
    code_blocks: List[Tuple[str, str]] = []

    for title, content in sections:
        title_lc = title.lower()
        if not any(k in title_lc for k in keys):
            continue
        titles.append(title)
        for line in content.splitlines():
            if len(snippets) >= 4:
                break
            raw = line.strip()
            if not raw or raw.startswith("```"):
                continue
            normalized = normalize_line(line)
            if not normalized:
                continue
            if normalized.startswith("<"):
                continue
            if normalized.lower() in {"bash", "shell", "python", "javascript", "typescript", "html"}:
                continue
            if len(normalized) < 3:
                continue
            if normalized:
                snippets.append(normalized)
        for block in extract_code_blocks(content, max_blocks=2):
            if len(code_blocks) >= 2:
                break
            code_blocks.append(block)
        if len(titles) >= 3:
            break
    return SectionMatch(titles=titles, snippet_lines=snippets, code_blocks=code_blocks)


def parse_github_repo(url: str) -> Optional[Tuple[str, str]]:
    pattern = re.compile(r"^https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+)")
    match = pattern.match(url.strip())
    if not match:
        return None
    owner = match.group(1).strip()
    repo = match.group(2).strip()
    repo = re.sub(r"\.git$", "", repo)
    if owner and repo:
        return owner, repo
    return None


def fetch_github_repo(owner: str, repo: str) -> RepoInfo:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    payload = fetch_json(url, extra_headers=GITHUB_API_HEADERS)
    if not isinstance(payload, dict):
        raise ToolNoteError("Unexpected GitHub repository payload.")
    return RepoInfo(
        owner=owner,
        repo=repo,
        html_url=payload.get("html_url", f"https://github.com/{owner}/{repo}"),
        description=(payload.get("description") or "").strip(),
        homepage=(payload.get("homepage") or "").strip(),
        language=(payload.get("language") or "").strip(),
        stars=payload.get("stargazers_count"),
        license_name=((payload.get("license") or {}).get("name") or "").strip(),
        topics=payload.get("topics") or [],
        updated_at=parse_date(payload.get("updated_at") or ""),
    )


def fetch_github_release(owner: str, repo: str) -> Optional[ReleaseInfo]:
    latest_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    try:
        payload = fetch_json(latest_url, extra_headers=GITHUB_API_HEADERS)
        if isinstance(payload, dict) and payload.get("html_url"):
            return ReleaseInfo(
                name=(payload.get("name") or "").strip(),
                tag_name=(payload.get("tag_name") or "").strip(),
                html_url=payload.get("html_url", ""),
                published_at=parse_date(payload.get("published_at") or ""),
            )
    except ToolNoteError:
        pass

    fallback_url = f"https://api.github.com/repos/{owner}/{repo}/releases?per_page=1"
    try:
        payload = fetch_json(fallback_url, extra_headers=GITHUB_API_HEADERS)
    except ToolNoteError:
        return None
    if isinstance(payload, list) and payload:
        item = payload[0]
        if isinstance(item, dict):
            return ReleaseInfo(
                name=(item.get("name") or "").strip(),
                tag_name=(item.get("tag_name") or "").strip(),
                html_url=item.get("html_url", ""),
                published_at=parse_date(item.get("published_at") or ""),
            )
    return None


def fetch_github_readme(owner: str, repo: str) -> Optional[ReadmeInfo]:
    readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        payload = fetch_json(readme_url, extra_headers=GITHUB_API_HEADERS)
    except ToolNoteError:
        return None
    if not isinstance(payload, dict):
        return None

    text = ""
    encoded = payload.get("content") or ""
    encoding = (payload.get("encoding") or "").lower()
    if encoded and encoding == "base64":
        try:
            text = base64.b64decode(encoded).decode("utf-8", errors="replace")
        except Exception:
            text = ""

    download_url = payload.get("download_url") or ""
    if not text and download_url:
        try:
            text = fetch_text(download_url)
        except ToolNoteError:
            text = ""

    if not text:
        return None

    return ReadmeInfo(
        text=text,
        html_url=(payload.get("html_url") or "").strip(),
        download_url=download_url.strip(),
        h1=extract_first_h1(text),
    )


def fetch_website_info(url: str) -> WebsiteInfo:
    html = fetch_text(url)
    parser = MetaParser()
    parser.feed(html)

    site_name = unescape(parser.meta.get("og:site_name", "")).strip()
    title = unescape(parser.meta.get("og:title", "") or parser.title).strip()
    description = unescape(
        parser.meta.get("description", "") or parser.meta.get("og:description", "")
    ).strip()

    github_candidates: List[str] = []
    for link in parser.links:
        if "github.com" not in link.lower():
            continue
        candidate = to_absolute_url(url, link)
        if parse_github_repo(candidate):
            github_candidates.append(candidate)

    github_url = choose_best_github_url(github_candidates, site_name, title, url)

    return WebsiteInfo(
        url=url,
        site_name=site_name,
        title=title,
        description=description,
        github_url=github_url,
    )


def tokenize_context(*texts: str) -> List[str]:
    tokens: List[str] = []
    for text in texts:
        for token in re.split(r"[^a-zA-Z0-9]+", text.lower()):
            if len(token) < 3:
                continue
            if token in {"www", "http", "https", "docs", "doc", "home"}:
                continue
            tokens.append(token)
    return tokens


def choose_best_github_url(candidates: List[str], site_name: str, title: str, page_url: str) -> str:
    if not candidates:
        return ""
    deduped: List[str] = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    netloc = urlparse(page_url).netloc
    context_tokens = tokenize_context(site_name, title, netloc)
    if not context_tokens:
        return deduped[0]

    best_url = deduped[0]
    best_score = -1
    for candidate in deduped:
        parsed = parse_github_repo(candidate)
        if not parsed:
            continue
        owner, repo = parsed
        owner_lc = owner.lower()
        repo_lc = repo.lower()
        combined = f"{owner_lc} {repo_lc}"
        score = 0
        for token in context_tokens:
            if token == repo_lc:
                score += 6
            elif token in repo_lc:
                score += 4
            if token == owner_lc:
                score += 3
            elif token in owner_lc:
                score += 2
            if token in combined:
                score += 1
        if score > best_score:
            best_score = score
            best_url = candidate
    return best_url


def to_absolute_url(base_url: str, candidate: str) -> str:
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    if candidate.startswith("//"):
        parsed = urlparse(base_url)
        return f"{parsed.scheme}:{candidate}"
    if candidate.startswith("/"):
        parsed = urlparse(base_url)
        return f"{parsed.scheme}://{parsed.netloc}{candidate}"
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    base_path = parsed.path.rsplit("/", 1)[0] if "/" in parsed.path else ""
    return f"{root}{base_path}/{candidate}".replace("//", "/").replace(":/", "://")


def build_overview_blurb(
    project_name: str,
    repo_info: Optional[RepoInfo],
    website_info: Optional[WebsiteInfo],
    readme_info: Optional[ReadmeInfo],
) -> str:
    if repo_info and repo_info.description:
        return repo_info.description
    if website_info and website_info.description:
        return website_info.description
    if readme_info and readme_info.text:
        for line in readme_info.text.splitlines():
            normalized = normalize_line(line)
            if normalized and len(normalized) > 20:
                return normalized
    return f"{project_name} 的可公开信息较少，当前为最小可用版记录。"


def choose_project_name(
    repo_info: Optional[RepoInfo], readme_info: Optional[ReadmeInfo], website_info: Optional[WebsiteInfo]
) -> str:
    if readme_info and readme_info.h1:
        return readme_info.h1
    if website_info and website_info.site_name:
        return website_info.site_name
    if website_info and website_info.title:
        return website_info.title
    if repo_info:
        return repo_info.repo
    if website_info:
        parsed = urlparse(website_info.url)
        return parsed.netloc or "tool-note"
    return "tool-note"


def format_section_lines(
    heading: str,
    match: Optional[SectionMatch],
    fallback_lines: List[str],
) -> List[str]:
    lines = [heading]
    if not match or (not match.titles and not match.snippet_lines and not match.code_blocks):
        lines.extend(fallback_lines)
        return lines

    if match.titles:
        lines.append(f"- 参考章节：{', '.join(match.titles)}")
    if match.snippet_lines:
        for item in match.snippet_lines[:4]:
            lines.append(f"- {item}")
    if match.code_blocks:
        lines.append("- 推荐命令示例：")
        for lang, code in match.code_blocks[:2]:
            lines.append(f"```{lang}")
            lines.append(code)
            lines.append("```")
    return lines


def build_markdown(
    *,
    url: str,
    note_date: str,
    project_name: str,
    repo_info: Optional[RepoInfo],
    release_info: Optional[ReleaseInfo],
    readme_info: Optional[ReadmeInfo],
    website_info: Optional[WebsiteInfo],
) -> str:
    tags = ["tools", "open-source"]
    if repo_info:
        tags.append("github")

    install_match = match_sections(readme_info.text, INSTALL_KEYWORDS) if readme_info else None
    first_run_match = match_sections(readme_info.text, FIRST_RUN_KEYWORDS) if readme_info else None
    usage_match = match_sections(readme_info.text, USAGE_KEYWORDS) if readme_info else None
    troubleshoot_match = match_sections(readme_info.text, TROUBLESHOOT_KEYWORDS) if readme_info else None

    overview = build_overview_blurb(project_name, repo_info, website_info, readme_info)

    lines: List[str] = [
        "---",
        f"date: {note_date}",
        "aliases: []",
        f"tags: [{', '.join(tags)}]",
        "---",
        "",
        f"# {project_name}",
        "",
        overview,
        "",
        "## 基本信息",
        f"- 输入链接：{url}",
        f"- 来源类型：{'GitHub 项目' if repo_info else '网页项目'}",
    ]

    if repo_info:
        lines.append(f"- GitHub 仓库：[{repo_info.owner}/{repo_info.repo}]({repo_info.html_url})")
        if repo_info.homepage:
            lines.append(f"- 官方网站：[{repo_info.homepage}]({repo_info.homepage})")
        if repo_info.language:
            lines.append(f"- 主要语言：{repo_info.language}")
        if repo_info.stars is not None:
            lines.append(f"- GitHub Stars：{repo_info.stars}")
        if repo_info.license_name:
            lines.append(f"- 开源协议：{repo_info.license_name}")
        if repo_info.updated_at:
            lines.append(f"- 最近更新：{repo_info.updated_at}")
    if release_info:
        release_label = release_info.name or release_info.tag_name or "latest"
        if release_info.html_url:
            lines.append(f"- 最新发布：[{release_label}]({release_info.html_url})")
        else:
            lines.append(f"- 最新发布：{release_label}")
        if release_info.published_at:
            lines.append(f"- 发布时间：{release_info.published_at}")
    if website_info:
        lines.append(f"- 网页标题：{website_info.title or '信息不足'}")
        if website_info.description:
            lines.append(f"- 页面描述：{website_info.description}")

    lines.extend(
        [
            "",
            "## 项目介绍（What it does）",
            "- 核心目标：基于公开资料自动汇总项目定位、安装和使用路径。",
        ]
    )
    if repo_info and repo_info.description:
        lines.append(f"- 仓库简介：{repo_info.description}")
    if repo_info and repo_info.topics:
        lines.append(f"- 主题标签：{', '.join(repo_info.topics[:8])}")
    if not repo_info and website_info and website_info.description:
        lines.append(f"- 官网摘要：{website_info.description}")
    if (not repo_info or not repo_info.description) and (not website_info or not website_info.description):
        lines.append("- 信息不足：未检索到稳定的项目摘要。")
        lines.append("- TODO: 进入项目文档页补充“适用场景”和“核心能力”。")

    lines.append("")
    lines.extend(
        format_section_lines(
            "## 安装方法（Installation）",
            install_match,
            [
                "- 信息不足：未在可解析章节中找到安装步骤。",
                "- TODO: 打开 README 或官网补充依赖、环境与安装命令。",
            ],
        )
    )

    lines.append("")
    lines.extend(
        format_section_lines(
            "## 首次使用（First run）",
            first_run_match,
            [
                "- 信息不足：未找到明确的首次启动流程。",
                "- TODO: 补充“首次运行命令 + 预期输出”。",
            ],
        )
    )

    lines.append("")
    lines.extend(
        format_section_lines(
            "## 后续使用（Daily usage）",
            usage_match,
            [
                "- 信息不足：未找到常用工作流或命令示例。",
                "- TODO: 补充日常操作流程（例如 update / run / config）。",
            ],
        )
    )

    lines.append("")
    lines.extend(
        format_section_lines(
            "## 常见问题与排错（Troubleshooting）",
            troubleshoot_match,
            [
                "- 信息不足：未找到 FAQ / troubleshooting 章节。",
                "- TODO: 增补常见报错、日志位置与修复路径。",
            ],
        )
    )

    references: List[str] = [url]
    if repo_info:
        references.append(repo_info.html_url)
    if readme_info:
        if readme_info.html_url:
            references.append(readme_info.html_url)
        elif readme_info.download_url:
            references.append(readme_info.download_url)
    if repo_info and repo_info.homepage:
        references.append(repo_info.homepage)
    if website_info and website_info.url not in references:
        references.append(website_info.url)

    deduped: List[str] = []
    seen = set()
    for ref in references:
        if not ref or ref in seen:
            continue
        seen.add(ref)
        deduped.append(ref)

    lines.extend(["", "## 参考来源（References）"])
    for ref in deduped:
        lines.append(f"- {ref}")

    lines.append("")
    return "\n".join(lines)


def resolve_output_path(vault_root: Path, output_dir: str, filename: str) -> Path:
    output_base = Path(output_dir)
    if not output_base.is_absolute():
        output_base = vault_root / output_base
    return output_base / f"{filename}.md"


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ToolNoteError(f"Invalid URL: {url}")


def write_markdown(path: Path, content: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ToolNoteError(f"File already exists and overwrite is disabled: {path}")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a 05_tools note from a URL.")
    parser.add_argument("--url", required=True, help="Project URL (GitHub or website)")
    parser.add_argument("--vault-root", required=True, help="Vault root path")
    parser.add_argument("--output-dir", default="05_tools", help="Output directory (default: 05_tools)")
    parser.add_argument("--date", default=date.today().strftime("%Y-%m-%d"), help="Note date (YYYY-MM-DD)")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()

    try:
        validate_url(args.url)
        note_date = parse_date(args.date)
        vault_root = Path(args.vault_root).expanduser().resolve()
        if not vault_root.exists():
            raise ToolNoteError(f"Vault root not found: {vault_root}")

        github_target = parse_github_repo(args.url)
        repo_info: Optional[RepoInfo] = None
        release_info: Optional[ReleaseInfo] = None
        readme_info: Optional[ReadmeInfo] = None
        website_info: Optional[WebsiteInfo] = None

        if github_target:
            owner, repo = github_target
            repo_info = fetch_github_repo(owner, repo)
            release_info = fetch_github_release(owner, repo)
            readme_info = fetch_github_readme(owner, repo)
            if repo_info.homepage:
                try:
                    website_info = fetch_website_info(repo_info.homepage)
                except ToolNoteError:
                    website_info = None
        else:
            website_info = fetch_website_info(args.url)
            if website_info.github_url:
                detected = parse_github_repo(website_info.github_url)
                if detected:
                    owner, repo = detected
                    repo_info = fetch_github_repo(owner, repo)
                    release_info = fetch_github_release(owner, repo)
                    readme_info = fetch_github_readme(owner, repo)

        project_name = choose_project_name(repo_info, readme_info, website_info)
        filename = sanitize_filename(project_name)
        output_path = resolve_output_path(vault_root, args.output_dir, filename)

        markdown = build_markdown(
            url=args.url,
            note_date=note_date,
            project_name=project_name,
            repo_info=repo_info,
            release_info=release_info,
            readme_info=readme_info,
            website_info=website_info,
        )
        write_markdown(output_path, markdown, overwrite=args.overwrite)
        rebuild_directory_page(output_path.parent)
        print(str(output_path.resolve()))
        return 0
    except ToolNoteError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
