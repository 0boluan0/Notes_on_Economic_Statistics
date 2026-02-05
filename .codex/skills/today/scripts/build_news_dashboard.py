#!/usr/bin/env python3
"""
Build module 1 static news dashboard for the daily note.

- Fetch 24h news from GDELT across categories
- Detect regions using keyword rules
- Generate a schematic world map SVG heat-style
- Output a markdown block for Module 1 to stdout
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "today-dashboard/1.0 (local)"
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

NEWS_CATEGORIES: Dict[str, str] = {
    "politics": "(politics OR government OR election OR congress)",
    "tech": "(technology OR software OR startup OR \"silicon valley\")",
    "finance": "(finance OR \"stock market\" OR economy OR banking)",
    "gov": "(\"federal government\" OR \"white house\" OR congress OR regulation)",
    "ai": "(\"artificial intelligence\" OR \"machine learning\" OR AI OR ChatGPT)",
    "intel": "(intelligence OR security OR military OR defense)",
}

REGION_KEYWORDS: Dict[str, List[str]] = {
    "EUROPE": [
        "nato",
        "eu",
        "european",
        "ukraine",
        "russia",
        "germany",
        "france",
        "uk",
        "britain",
        "poland",
    ],
    "MENA": [
        "iran",
        "israel",
        "saudi",
        "syria",
        "iraq",
        "gaza",
        "lebanon",
        "yemen",
        "houthi",
        "middle east",
    ],
    "APAC": [
        "china",
        "taiwan",
        "japan",
        "korea",
        "indo-pacific",
        "south china sea",
        "asean",
        "philippines",
    ],
    "AMERICAS": ["us", "america", "canada", "mexico", "brazil", "venezuela", "latin"],
    "AFRICA": ["africa", "sahel", "niger", "sudan", "ethiopia", "somalia"],
}

REGION_ORDER = ["AMERICAS", "EUROPE", "MENA", "AFRICA", "APAC"]
REGION_LABELS = {
    "AMERICAS": "美洲 / Americas",
    "EUROPE": "欧洲 / Europe",
    "MENA": "中东北非 / MENA",
    "AFRICA": "非洲 / Africa",
    "APAC": "亚太 / APAC",
    "UNKNOWN": "其他 / Unknown",
}

MAX_REGION_ITEMS = 3
MAX_TECH_ITEMS = 3
MAX_INTEL_ITEMS = 3

LEGEND_STEPS: List[Tuple[str, str]] = [
    ("0", "#f2f2f2"),
    ("1-2", "#d6e8ff"),
    ("3-5", "#a8ccff"),
    ("6-10", "#6ea7ff"),
    ("11-20", "#2f7bff"),
    ("21+", "#0b4ec2"),
]


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    timestamp: dt.datetime
    category: str
    region: Optional[str]

    def local_time(self, tzinfo: dt.tzinfo) -> str:
        return self.timestamp.astimezone(tzinfo).strftime("%Y-%m-%d %H:%M")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static news dashboard module.")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Attempt machine translation for Chinese lines via LibreTranslate (optional).",
    )
    return parser.parse_args()


def parse_date(raw: Optional[str]) -> dt.date:
    if not raw:
        return dt.date.today()
    return dt.datetime.strptime(raw, "%Y-%m-%d").date()


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def parse_gdelt_date(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    match = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$", value)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone.utc)
    try:
        cleaned = value.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    except ValueError:
        return None


def fetch_json(url: str) -> Optional[dict]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=20) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
            if not payload.strip().startswith("{"):
                return None
            return json.loads(payload)
    except Exception as exc:
        print(f"[news-dashboard] fetch failed: {exc}", file=sys.stderr)
        return None


def fetch_category(category: str, max_records: int = 80) -> List[NewsItem]:
    query = NEWS_CATEGORIES[category]
    full_query = f"{query} sourcelang:english"
    params = {
        "query": full_query,
        "timespan": "1d",
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        "sort": "date",
    }
    url = f"{GDELT_ENDPOINT}?{urlencode(params)}"
    data = fetch_json(url)
    if not data or "articles" not in data:
        return []

    items: List[NewsItem] = []
    for article in data.get("articles", []):
        raw_title = article.get("title") or ""
        title = re.sub(r"\s+", " ", raw_title).strip()
        link = (article.get("url") or "").strip()
        seendate = article.get("seendate") or ""
        domain = (article.get("domain") or "").strip()
        if not title or not link:
            continue
        timestamp = parse_gdelt_date(seendate)
        if not timestamp:
            continue
        source = domain or urlparse(link).netloc or "Unknown"
        items.append(
            NewsItem(
                title=title,
                link=link,
                source=source,
                timestamp=timestamp,
                category=category,
                region=None,
            )
        )
    return items


def detect_region(text: str) -> Optional[str]:
    lower = text.lower()
    for region, keywords in REGION_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return region
    return None


def dedupe_items(items: Iterable[NewsItem]) -> List[NewsItem]:
    seen: set[str] = set()
    result: List[NewsItem] = []
    for item in items:
        key = item.link or item.title.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def filter_window(
    items: Iterable[NewsItem], start: dt.datetime, end: dt.datetime
) -> List[NewsItem]:
    filtered = [item for item in items if start <= item.timestamp <= end]
    filtered.sort(key=lambda x: x.timestamp, reverse=True)
    return filtered


def color_for_count(count: int) -> str:
    if count <= 0:
        return LEGEND_STEPS[0][1]
    if count <= 2:
        return LEGEND_STEPS[1][1]
    if count <= 5:
        return LEGEND_STEPS[2][1]
    if count <= 10:
        return LEGEND_STEPS[3][1]
    if count <= 20:
        return LEGEND_STEPS[4][1]
    return LEGEND_STEPS[5][1]


def text_color(hex_color: str) -> str:
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    return "#0f172a" if luminance > 0.6 else "#ffffff"


def build_map_svg(
    counts: Dict[str, int],
    window_text: str,
    output_path: Path,
) -> None:
    width = 900
    height = 520
    region_shapes = {
        "AMERICAS": {"label": "Americas 美洲", "x": 80, "y": 170, "w": 210, "h": 190},
        "EUROPE": {"label": "Europe 欧洲", "x": 340, "y": 120, "w": 140, "h": 90},
        "MENA": {"label": "MENA 中东北非", "x": 370, "y": 230, "w": 150, "h": 80},
        "AFRICA": {"label": "Africa 非洲", "x": 340, "y": 320, "w": 150, "h": 140},
        "APAC": {"label": "APAC 亚太", "x": 540, "y": 180, "w": 250, "h": 190},
    }

    svg_lines: List[str] = []
    svg_lines.append(
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )
    svg_lines.append("<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
    svg_lines.append(
        "<text x=\"40\" y=\"46\" font-size=\"20\" font-family=\"Arial, sans-serif\" fill=\"#0f172a\">全球态势地图 | Global Situation Map</text>"
    )
    svg_lines.append(
        f"<text x=\"40\" y=\"72\" font-size=\"12\" font-family=\"Arial, sans-serif\" fill=\"#475569\">时间窗口: {window_text}</text>"
    )
    svg_lines.append(
        "<text x=\"40\" y=\"490\" font-size=\"11\" font-family=\"Arial, sans-serif\" fill=\"#64748b\">示意区域块，按新闻数量着色</text>"
    )
    svg_lines.append(
        "<ellipse cx=\"450\" cy=\"285\" rx=\"390\" ry=\"190\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>"
    )

    for region_id, shape in region_shapes.items():
        count = counts.get(region_id, 0)
        fill = color_for_count(count)
        label = shape["label"]
        x = shape["x"]
        y = shape["y"]
        w = shape["w"]
        h = shape["h"]
        text_fill = text_color(fill)
        svg_lines.append(
            f"<rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" rx=\"16\" fill=\"{fill}\" stroke=\"#94a3b8\" stroke-width=\"1\"/>"
        )
        svg_lines.append(
            f"<text x=\"{x + 12}\" y=\"{y + 28}\" font-size=\"13\" font-family=\"Arial, sans-serif\" fill=\"{text_fill}\">{label}</text>"
        )
        svg_lines.append(
            f"<text x=\"{x + 12}\" y=\"{y + 52}\" font-size=\"12\" font-family=\"Arial, sans-serif\" fill=\"{text_fill}\">新闻数: {count}</text>"
        )

    legend_x = 640
    legend_y = 405
    svg_lines.append(
        "<text x=\"640\" y=\"390\" font-size=\"12\" font-family=\"Arial, sans-serif\" fill=\"#475569\">热度图例 | Counts</text>"
    )
    for idx, (label, color) in enumerate(LEGEND_STEPS):
        y = legend_y + idx * 18
        svg_lines.append(
            f"<rect x=\"{legend_x}\" y=\"{y}\" width=\"14\" height=\"14\" fill=\"{color}\" stroke=\"#94a3b8\" stroke-width=\"0.5\"/>"
        )
        svg_lines.append(
            f"<text x=\"{legend_x + 22}\" y=\"{y + 12}\" font-size=\"11\" font-family=\"Arial, sans-serif\" fill=\"#475569\">{label}</text>"
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")


def format_item_lines(
    item: NewsItem,
    tzinfo: dt.tzinfo,
    tz_label: str,
    translate_enabled: bool,
    translator,
) -> List[str]:
    title_en = item.title
    title_zh = translator(title_en) if translate_enabled else title_en
    time_str = item.local_time(tzinfo)
    lines = [
        f"- 中文：{title_zh}",
        f"  EN: {title_en}",
        f"  来源：{item.source}（{time_str}, {tz_label}）",
    ]
    return lines


def build_markdown(
    map_filename: str,
    window_text: str,
    tz_label: str,
    tzinfo: dt.tzinfo,
    region_items: Dict[str, List[NewsItem]],
    tech_items: List[NewsItem],
    intel_items: List[NewsItem],
    translate_enabled: bool,
    translator,
) -> str:
    lines: List[str] = []
    lines.append("## 模块一｜全球态势静态快照（中英双语）")
    lines.append(f"> 时间窗口：{window_text} ({tz_label})")
    if not translate_enabled:
        lines.append("> 注：中文行默认使用英文标题，可手动润色或配置 LibreTranslate 翻译服务。")
    lines.append("")
    lines.append(f"![[98_attachment/dashboards/{map_filename}]]")
    lines.append("")
    lines.append("### 区域新闻（Regions）")

    for region in REGION_ORDER:
        lines.append(f"#### {REGION_LABELS[region]}")
        items = region_items.get(region, [])[:MAX_REGION_ITEMS]
        if not items:
            lines.extend(
                [
                    "- 中文：暂无符合条件的新闻",
                    "  EN: No items in window",
                    "  来源：—",
                ]
            )
        else:
            for item in items:
                lines.extend(format_item_lines(item, tzinfo, tz_label, translate_enabled, translator))

    unknown_items = region_items.get("UNKNOWN", [])[:MAX_REGION_ITEMS]
    if unknown_items:
        lines.append(f"#### {REGION_LABELS['UNKNOWN']}")
        for item in unknown_items:
            lines.extend(format_item_lines(item, tzinfo, tz_label, translate_enabled, translator))

    lines.append("")
    lines.append("### 全球科技（Global Technology）")
    if not tech_items:
        lines.extend(
            [
                "- 中文：暂无符合条件的新闻",
                "  EN: No items in window",
                "  来源：—",
            ]
        )
    else:
        for item in tech_items[:MAX_TECH_ITEMS]:
            lines.extend(format_item_lines(item, tzinfo, tz_label, translate_enabled, translator))

    lines.append("")
    lines.append("### 全球情报/地缘政经（Global Intelligence & Geopolitics）")
    if not intel_items:
        lines.extend(
            [
                "- 中文：暂无符合条件的新闻",
                "  EN: No items in window",
                "  来源：—",
            ]
        )
    else:
        for item in intel_items[:MAX_INTEL_ITEMS]:
            lines.extend(format_item_lines(item, tzinfo, tz_label, translate_enabled, translator))

    return "\n".join(lines) + "\n"


def make_translator(enabled: bool):
    if not enabled:
        return lambda text: text

    service_url = os.environ.get("LIBRETRANSLATE_URL", "https://libretranslate.de/translate")

    cache: Dict[str, str] = {}

    def translate(text: str) -> str:
        if not text:
            return text
        if text in cache:
            return cache[text]
        try:
            payload = json.dumps(
                {
                    "q": text,
                    "source": "en",
                    "target": "zh",
                    "format": "text",
                }
            ).encode("utf-8")
            req = Request(
                service_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            )
            with urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                translated = data.get("translatedText") or text
                cache[text] = translated
                return translated
        except Exception:
            return text

    return translate


def main() -> int:
    args = parse_args()
    target_date = parse_date(args.date)
    now = now_local()
    window_end = now
    window_start = now - dt.timedelta(hours=24)
    tz_label = now.tzname() or "Local"
    window_text = f"{window_start:%Y-%m-%d %H:%M} ~ {window_end:%Y-%m-%d %H:%M}"

    items_all: List[NewsItem] = []
    for category in NEWS_CATEGORIES:
        items_all.extend(fetch_category(category))

    for item in items_all:
        item.region = detect_region(item.title)

    items_all = filter_window(items_all, window_start, window_end)
    items_all = dedupe_items(items_all)

    region_items: Dict[str, List[NewsItem]] = {region: [] for region in REGION_ORDER}
    region_items["UNKNOWN"] = []

    for item in items_all:
        region = item.region or "UNKNOWN"
        if region in region_items:
            region_items[region].append(item)
        else:
            region_items["UNKNOWN"].append(item)

    for items in region_items.values():
        items.sort(key=lambda x: x.timestamp, reverse=True)

    tech_items = [item for item in items_all if item.category == "tech"]
    intel_items = [
        item for item in items_all if item.category in {"intel", "gov", "politics"}
    ]

    tech_items = dedupe_items(tech_items)
    intel_items = dedupe_items(intel_items)

    counts = {region: len(region_items.get(region, [])) for region in REGION_ORDER}

    vault_root = Path(__file__).resolve().parents[4]
    output_dir = vault_root / "98_attachment" / "dashboards"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_filename = f"{target_date:%Y-%m-%d}-map.svg"
    map_path = output_dir / map_filename

    build_map_svg(counts, window_text, map_path)

    translator = make_translator(args.translate)
    markdown = build_markdown(
        map_filename,
        window_text,
        tz_label,
        now.tzinfo or dt.timezone.utc,
        region_items,
        tech_items,
        intel_items,
        args.translate,
        translator,
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
