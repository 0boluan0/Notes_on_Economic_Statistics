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
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
TOPOJSON_PATH = ASSETS_DIR / "countries-110m.json"
FETCH_ERROR_COUNT = 0

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
MAX_FINANCE_ITEMS = 3
MAX_INTEL_ITEMS = 3

LEGEND_STEPS: List[Tuple[str, str]] = [
    ("0", "#f7f1e6"),
    ("1-2", "#f4d8c5"),
    ("3-5", "#f2b58f"),
    ("6-10", "#ee8a5b"),
    ("11-20", "#e45b3b"),
    ("21+", "#c7372f"),
]

KEYWORD_STOPWORDS = {
    "after",
    "amid",
    "and",
    "among",
    "add",
    "about",
    "accuse",
    "accuses",
    "announces",
    "announced",
    "as",
    "at",
    "back",
    "by",
    "before",
    "between",
    "billion",
    "breaks",
    "calls",
    "can",
    "could",
    "court",
    "claims",
    "day",
    "deal",
    "despite",
    "for",
    "from",
    "gets",
    "global",
    "group",
    "government",
    "had",
    "has",
    "have",
    "his",
    "its",
    "into",
    "last",
    "latest",
    "market",
    "more",
    "new",
    "not",
    "officials",
    "over",
    "plan",
    "plans",
    "report",
    "reports",
    "said",
    "says",
    "set",
    "says",
    "shares",
    "state",
    "stocks",
    "talks",
    "than",
    "that",
    "the",
    "their",
    "this",
    "three",
    "time",
    "to",
    "today",
    "under",
    "up",
    "using",
    "war",
    "week",
    "what",
    "while",
    "with",
    "will",
    "wins",
    "year",
    "years",
}

KEYWORD_KEEP_UPPER = {"ai", "us", "uk", "eu", "nato"}


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
    global FETCH_ERROR_COUNT
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=20) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
            if not payload.strip().startswith("{"):
                return None
            return json.loads(payload)
    except Exception as exc:
        FETCH_ERROR_COUNT += 1
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
        title_key = re.sub(r"[^a-z0-9]+", "", item.title.lower())
        key = item.link or title_key
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def load_cache(path: Path) -> List[NewsItem]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items: List[NewsItem] = []
    for row in payload:
        try:
            ts = dt.datetime.fromisoformat(row["timestamp"])
        except (KeyError, ValueError):
            continue
        items.append(
            NewsItem(
                title=row.get("title", ""),
                link=row.get("link", ""),
                source=row.get("source", ""),
                timestamp=ts,
                category=row.get("category", ""),
                region=row.get("region"),
            )
        )
    return items


def save_cache(path: Path, items: List[NewsItem]) -> None:
    payload = [
        {
            "title": item.title,
            "link": item.link,
            "source": item.source,
            "timestamp": item.timestamp.isoformat(),
            "category": item.category,
            "region": item.region,
        }
        for item in items
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def filter_window(
    items: Iterable[NewsItem], start: dt.datetime, end: dt.datetime
) -> List[NewsItem]:
    filtered = [item for item in items if start <= item.timestamp <= end]
    filtered.sort(key=lambda x: x.timestamp, reverse=True)
    return filtered


def extract_keywords(items: Iterable[NewsItem]) -> str:
    counts: Dict[str, int] = {}
    for item in items:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9']+", item.title)
        for token in tokens:
            word = token.strip("'").lower()
            if not word or word in KEYWORD_STOPWORDS:
                continue
            if len(word) < 3 and word not in KEYWORD_KEEP_UPPER:
                continue
            counts[word] = counts.get(word, 0) + 1

    if not counts:
        return "—"

    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    formatted = []
    for word, _ in top:
        if word in KEYWORD_KEEP_UPPER:
            formatted.append(word.upper())
        else:
            formatted.append(word)

    joined = ", ".join(formatted)
    if len(joined) > 36:
        joined = joined[:33].rstrip(", ") + "…"
    return joined


def classify_region(lon: float, lat: float) -> Optional[str]:
    if lat < -55:
        return None
    if lon < -30:
        return "AMERICAS"
    if lon >= 60:
        return "APAC"
    if lat >= 35:
        return "EUROPE"
    if lat >= 12:
        return "MENA"
    return "AFRICA"


def load_topojson(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def decode_arcs(data: dict) -> List[List[Tuple[float, float]]]:
    transform = data.get("transform")
    arcs = data.get("arcs", [])
    if not transform:
        return []
    scale = transform["scale"]
    translate = transform["translate"]

    decoded: List[List[Tuple[float, float]]] = []
    for arc in arcs:
        x = 0
        y = 0
        coords: List[Tuple[float, float]] = []
        for dx, dy in arc:
            x += dx
            y += dy
            lon = x * scale[0] + translate[0]
            lat = y * scale[1] + translate[1]
            coords.append((lon, lat))
        decoded.append(coords)
    return decoded


def arc_by_index(arcs: List[List[Tuple[float, float]]], idx: int) -> List[Tuple[float, float]]:
    if idx >= 0:
        return arcs[idx]
    return list(reversed(arcs[~idx]))


def arcs_to_coords(
    arcs: List[List[Tuple[float, float]]], arc_indices: List[int]
) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for idx in arc_indices:
        arc = arc_by_index(arcs, idx)
        if coords:
            coords.extend(arc[1:])
        else:
            coords.extend(arc)
    return coords


def geometry_rings(geom: dict) -> List[List[List[int]]]:
    if geom["type"] == "Polygon":
        return [geom["arcs"]]
    if geom["type"] == "MultiPolygon":
        return geom["arcs"]
    return []


def centroid_from_coords(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not coords:
        return (0.0, 0.0)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


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


def build_fallback_svg(
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


def build_map_svg(
    counts: Dict[str, int],
    keywords: Dict[str, str],
    window_text: str,
    output_path: Path,
) -> None:
    topo = load_topojson(TOPOJSON_PATH)
    if not topo:
        build_fallback_svg(counts, window_text, output_path)
        return

    arcs = decode_arcs(topo)
    countries = topo.get("objects", {}).get("countries", {}).get("geometries", [])
    if not arcs or not countries:
        build_fallback_svg(counts, window_text, output_path)
        return

    width = 1000
    height = 520
    map_x = 40
    map_y = 90
    map_w = 920
    map_h = 360

    def proj(lon: float, lat: float) -> Tuple[float, float]:
        x = (lon + 180.0) / 360.0 * map_w + map_x
        y = (90.0 - lat) / 180.0 * map_h + map_y
        return x, y

    svg_lines: List[str] = []
    svg_lines.append(
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )
    svg_lines.append("<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
    svg_lines.append(
        "<text x=\"40\" y=\"46\" font-size=\"20\" font-family=\"Arial, sans-serif\" fill=\"#1f2937\">全球态势地图 | Global Situation Map</text>"
    )
    svg_lines.append(
        f"<text x=\"40\" y=\"72\" font-size=\"12\" font-family=\"Arial, sans-serif\" fill=\"#6b7280\">时间窗口: {window_text}</text>"
    )

    svg_lines.append("<g id=\"countries\" stroke=\"#ffffff\" stroke-width=\"0.2\">")
    for geom in countries:
        rings = geometry_rings(geom)
        if not rings:
            continue

        longest: List[Tuple[float, float]] = []
        for polygon in rings:
            if not polygon:
                continue
            coords = arcs_to_coords(arcs, polygon[0])
            if len(coords) > len(longest):
                longest = coords

        lon, lat = centroid_from_coords(longest)
        region = classify_region(lon, lat)
        fill = color_for_count(counts.get(region, 0)) if region else "#f1f5f9"

        for polygon in rings:
            path_parts: List[str] = []
            for ring in polygon:
                coords = arcs_to_coords(arcs, ring)
                if not coords:
                    continue
                for idx, (lon_pt, lat_pt) in enumerate(coords):
                    x, y = proj(lon_pt, lat_pt)
                    if idx == 0:
                        path_parts.append(f"M{x:.2f},{y:.2f}")
                    else:
                        path_parts.append(f"L{x:.2f},{y:.2f}")
                path_parts.append("Z")
            if not path_parts:
                continue
            svg_lines.append(
                f"<path d=\"{' '.join(path_parts)}\" fill=\"{fill}\" fill-opacity=\"0.85\" fill-rule=\"evenodd\"/>"
            )
    svg_lines.append("</g>")

    label_positions = {
        "AMERICAS": (-105, 20),
        "EUROPE": (10, 58),
        "MENA": (25, 24),
        "AFRICA": (20, -10),
        "APAC": (120, 15),
    }
    svg_lines.append("<g font-family=\"Arial, sans-serif\" fill=\"#111827\" font-size=\"12\" text-anchor=\"middle\">")
    for region in REGION_ORDER:
        lon, lat = label_positions[region]
        x, y = proj(lon, lat)
        label = REGION_LABELS.get(region, region)
        count = counts.get(region, 0)
        svg_lines.append(
            f"<text x=\"{x:.2f}\" y=\"{y:.2f}\">{label} | {count}</text>"
        )
        svg_lines.append(
            f"<text x=\"{x:.2f}\" y=\"{y+14:.2f}\" font-size=\"10\" fill=\"#374151\">{keywords.get(region, '—')}</text>"
        )
    svg_lines.append("</g>")

    legend_x = 730
    legend_y = 452
    svg_lines.append("<g font-family=\"Arial, sans-serif\" fill=\"#374151\" font-size=\"11\">")
    svg_lines.append(f"<text x=\"{legend_x}\" y=\"{legend_y}\">热度图例 | Counts</text>")
    for idx, (label, color) in enumerate(LEGEND_STEPS):
        y = legend_y + 10 + idx * 16
        svg_lines.append(
            f"<rect x=\"{legend_x}\" y=\"{y}\" width=\"12\" height=\"12\" fill=\"{color}\" stroke=\"#9ca3af\" stroke-width=\"0.4\"/>"
        )
        svg_lines.append(f"<text x=\"{legend_x+18}\" y=\"{y+10}\">{label}</text>")
    svg_lines.append("</g>")

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")


def allocate_rows(counts: Dict[str, int]) -> Dict[str, int]:
    total = sum(counts.values())
    if total <= 0:
        return {region: 0 for region in REGION_ORDER}

    total_rows = min(8, max(4, total // 50 + 4))
    raw = {region: (counts[region] / total) * total_rows for region in REGION_ORDER}
    rows: Dict[str, int] = {}
    for region in REGION_ORDER:
        if counts[region] <= 0:
            rows[region] = 0
        else:
            rows[region] = max(1, int(raw[region]))

    current = sum(rows.values())
    if current < total_rows:
        order = sorted(REGION_ORDER, key=lambda r: raw[r] - int(raw[r]), reverse=True)
        idx = 0
        while current < total_rows:
            region = order[idx % len(order)]
            if counts[region] > 0:
                rows[region] += 1
                current += 1
            idx += 1
    elif current > total_rows:
        order = sorted(REGION_ORDER, key=lambda r: raw[r] - int(raw[r]))
        idx = 0
        while current > total_rows and idx < len(order):
            region = order[idx]
            if rows[region] > 1:
                rows[region] -= 1
                current -= 1
            idx += 1

    return rows


def summarize_item(item: NewsItem, translate_enabled: bool, translator) -> str:
    title_en = item.title
    if translate_enabled:
        title_zh = translator(title_en)
        return title_zh
    return title_en


def unique_by_title(items: List[NewsItem]) -> List[NewsItem]:
    seen: set[str] = set()
    result: List[NewsItem] = []
    for item in items:
        key = re.sub(r"[^a-z0-9]+", "", item.title.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_markdown(
    map_filename: str,
    window_text: str,
    tz_label: str,
    tzinfo: dt.tzinfo,
    region_items: Dict[str, List[NewsItem]],
    finance_items: List[NewsItem],
    tech_items: List[NewsItem],
    translate_enabled: bool,
    translator,
    data_status_note: Optional[str] = None,
    no_data_reason: Optional[str] = None,
) -> str:
    lines: List[str] = []
    lines.append("## 模块一｜全球态势静态快照（中文为主）")
    lines.append(f"> 时间窗口：{window_text} ({tz_label})")
    lines.append("> 注：表格为摘要，不含来源。")
    if data_status_note:
        lines.append(data_status_note)
    lines.append("")
    lines.append(f"![[98_attachment/dashboards/{map_filename}]]")
    lines.append("")
    lines.append("### 今日要点（按热度分配篇幅）")
    lines.append("| 区域 | 热度 | 发生了什么 |")
    lines.append("| --- | --- | --- |")

    counts = {region: len(region_items.get(region, [])) for region in REGION_ORDER}
    rows = allocate_rows(counts)
    any_row = False

    for region in REGION_ORDER:
        take = rows.get(region, 0)
        if take <= 0:
            continue
        items = unique_by_title(region_items.get(region, []))[:take]
        if not items:
            continue
        for idx, item in enumerate(items):
            label = REGION_LABELS[region] if idx == 0 else ""
            summary = summarize_item(item, translate_enabled, translator)
            lines.append(f"| {label} | {counts[region]} | {summary} |")
            any_row = True

    if not any_row:
        lines.append(f"| 全局 | 0 | {no_data_reason or '暂无显著新闻'} |")

    lines.append("")
    lines.append("### 金融要点")
    finance_list = unique_by_title(finance_items)[:MAX_FINANCE_ITEMS]
    if finance_list:
        for item in finance_list:
            summary = summarize_item(item, translate_enabled, translator)
            lines.append(f"- {summary}")
    else:
        lines.append(f"- {no_data_reason or '暂无显著新闻'}")

    lines.append("")
    lines.append("### 科技要点")
    tech_list = unique_by_title(tech_items)[:MAX_TECH_ITEMS]
    if tech_list:
        for item in tech_list:
            summary = summarize_item(item, translate_enabled, translator)
            lines.append(f"- {summary}")
    else:
        lines.append(f"- {no_data_reason or '暂无显著新闻'}")

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

    counts = {region: len(region_items.get(region, [])) for region in REGION_ORDER}
    region_keywords = {
        region: extract_keywords(region_items.get(region, [])) for region in REGION_ORDER
    }

    vault_root = Path(__file__).resolve().parents[4]
    output_dir = vault_root / "98_attachment" / "dashboards"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{target_date:%Y-%m-%d}-news.json"
    data_status_note: Optional[str] = None
    no_data_reason: Optional[str] = None
    if not items_all:
        cached = load_cache(cache_path)
        if cached:
            items_all = cached
            items_all = dedupe_items(items_all)
            data_status_note = f"> 数据状态：在线抓取失败，使用本地缓存（{target_date:%Y-%m-%d}）。"
            for item in items_all:
                if not item.region:
                    item.region = detect_region(item.title)
            region_items = {region: [] for region in REGION_ORDER}
            region_items["UNKNOWN"] = []
            for item in items_all:
                region = item.region or "UNKNOWN"
                if region in region_items:
                    region_items[region].append(item)
                else:
                    region_items["UNKNOWN"].append(item)
            for items in region_items.values():
                items.sort(key=lambda x: x.timestamp, reverse=True)
            counts = {region: len(region_items.get(region, [])) for region in REGION_ORDER}
            region_keywords = {
                region: extract_keywords(region_items.get(region, [])) for region in REGION_ORDER
            }
        elif FETCH_ERROR_COUNT > 0:
            no_data_reason = "网络受限，未抓取到有效新闻（可提权重试）"
            data_status_note = "> 数据状态：在线抓取失败，且本地无当日缓存。"
    elif items_all:
        save_cache(cache_path, items_all)
    finance_items = [item for item in items_all if item.category == "finance"]
    finance_items.sort(key=lambda x: x.timestamp, reverse=True)
    tech_items = [item for item in items_all if item.category in {"tech", "ai"}]
    tech_items.sort(key=lambda x: x.timestamp, reverse=True)
    map_filename = f"{target_date:%Y-%m-%d}-map.svg"
    map_path = output_dir / map_filename

    build_map_svg(counts, region_keywords, window_text, map_path)

    translator = make_translator(args.translate)
    markdown = build_markdown(
        map_filename,
        window_text,
        tz_label,
        now.tzinfo or dt.timezone.utc,
        region_items,
        finance_items,
        tech_items,
        args.translate,
        translator,
        data_status_note=data_status_note,
        no_data_reason=no_data_reason,
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
