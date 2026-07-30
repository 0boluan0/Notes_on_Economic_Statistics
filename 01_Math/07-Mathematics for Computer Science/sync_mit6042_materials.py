#!/usr/bin/env python3
"""Synchronize the useful MIT 6.042J Spring 2015 OCW course assets.

The official legacy OCW archive contains the complete course except for the
video files.  This script keeps the learning assets (PDFs, captions, exercise
content, and exercise figures), discards the offline-site runtime, and builds
machine-readable inventories used by the course notes.

Required packages:
    python3 -m pip install beautifulsoup4 markdownify pypdf

Typical use with an already extracted archive:
    python3 sync_mit6042_materials.py \
        --source-dir /tmp/mit6042-full/6-042j-spring-2015 \
        --archive /tmp/6-042j-spring-2015-full.zip

When neither --source-dir nor --archive is supplied, the script downloads the
official archive to a temporary directory and verifies its SHA-256 digest.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "Missing dependency. Run: python3 -m pip install "
        "beautifulsoup4 markdownify pypdf"
    ) from exc


COURSE_ID = "MIT 6.042J Mathematics for Computer Science, Spring 2015"
COURSE_URL = (
    "https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-"
    "spring-2015/"
)
LEGACY_COURSE_URL = (
    "https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/"
    "6-042j-mathematics-for-computer-science-spring-2015/"
)
ARCHIVE_URL = (
    "https://ocw.mit.edu/ans15436/ZipForEndUsers/6/6-042j-spring-2015/"
    "6-042j-spring-2015.zip"
)
ARCHIVE_SHA256 = "2d248850dbe0bf00412ca6091d8214c04e33a02d32b33d9dee9d56fbbb2a4967"

EXPECTED = {
    "sessions": 35,
    "pdf": 311,
    "readings": 35,
    "books": 1,
    "slides": 113,
    "slides_s15": 63,
    "slides_s16": 50,
    "transcripts": 111,
    "captions": 111,
    "in_class": 35,
    "problem_sets": 12,
    "exams": 4,
    "courseware_blocks": 284,
    "exercise_blocks": 171,
    "prompts": 376,
    "video_blocks": 111,
    "slide_only_blocks": 2,
    "courseware_images": 17,
    "coverage_rows": 596,
    "coverage_online": 376,
    "coverage_problem_sets": 38,
    "coverage_in_class": 153,
    "coverage_exams": 29,
}

OUTPUT_DIRS = (
    "01_Session_Readings",
    "02_Lecture_Slides",
    "03_Video_Transcripts",
    "04_Captions",
    "05_In_Class_Questions",
    "06_Problem_Sets",
    "07_Exams",
    "08_Courseware_Exercises",
    "09_Courseware_Images",
    "99_Books",
)

UNIT_INFO = {
    "proofs": (1, "Unit 1: Proofs", "01_Proofs.md", 0),
    "structures": (2, "Unit 2: Structures", "02_Structures.md", 11),
    "counting": (3, "Unit 3: Counting", "03_Counting.md", 22),
    "probability": (4, "Unit 4: Probability", "04_Probability.md", 27),
}

PSET_SESSION_RANGES = {
    1: (1, 4),
    2: (5, 6),
    3: (7, 8),
    4: (9, 11),
    5: (12, 14),
    6: (15, 16),
    7: (17, 19),
    8: (20, 22),
    9: (23, 24),
    10: (25, 27),
    11: (28, 29),
    12: (30, 32),
}

EXAM_INFO = {
    "MIT6_042JS15_midterm1.pdf": ("Midterm 1", 8, "01_Proofs.md"),
    "MIT6_042JS15_midterm2.pdf": ("Midterm 2", 16, "02_Structures.md"),
    "MIT6_042JS15_midterm3.pdf": ("Midterm 3", 24, "03_Counting.md"),
    "MIT6_042JS15_finalexam.pdf": (
        "Final Exam",
        35,
        "05_Review and exam roadmap.md",
    ),
}

MANIFEST_FIELDS = (
    "asset_id",
    "category",
    "session",
    "unit",
    "block",
    "resource_title",
    "original_basename",
    "local_path",
    "source_relative_path",
    "official_url",
    "provenance",
    "year",
    "file_format",
    "pages",
    "bytes",
    "sha256",
)

COVERAGE_FIELDS = (
    "coverage_id",
    "category",
    "session",
    "unit",
    "source_file",
    "source_page",
    "problem_number",
    "prompt_id",
    "official_answer_available",
    "note_file",
    "note_anchor",
    "solution_status",
    "verification_method",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def slugify(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"\[optional\]", "", value, flags=re.I)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "exercise"


def unit_for_session(session: int) -> tuple[int, str, str]:
    for _slug, (unit_no, unit_title, note_file, offset) in UNIT_INFO.items():
        upper = {0: 11, 11: 22, 22: 27, 27: 35}[offset]
        if offset < session <= upper:
            return unit_no, unit_title, note_file
    raise ValueError(f"Session outside course: {session}")


def official_asset_url(relative_to_contents: Path) -> str:
    return LEGACY_COURSE_URL + relative_to_contents.as_posix()


def archive_digest_ok(path: Path) -> bool:
    actual = sha256(path)
    if actual != ARCHIVE_SHA256:
        raise ValueError(
            f"Archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
        )
    return True


def find_course_root(extracted_root: Path) -> Path:
    direct = extracted_root / "6-042j-spring-2015"
    candidates = [direct, extracted_root]
    candidates.extend(extracted_root.glob("**/6-042j-spring-2015"))
    for candidate in candidates:
        if (candidate / "contents/resource-index/index.htm").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the extracted 6-042j-spring-2015 root")


@contextlib.contextmanager
def course_source(args: argparse.Namespace) -> Iterator[Path]:
    if args.archive:
        archive_digest_ok(args.archive.resolve())

    if args.source_dir:
        source = args.source_dir.resolve()
        if not (source / "contents/resource-index/index.htm").is_file():
            raise FileNotFoundError(f"Invalid course source directory: {source}")
        yield source
        return

    with tempfile.TemporaryDirectory(prefix="mit6042-sync-") as temp_name:
        temp = Path(temp_name)
        archive = args.archive.resolve() if args.archive else temp / "course.zip"
        if not args.archive:
            print(f"Downloading official archive: {ARCHIVE_URL}")
            urllib.request.urlretrieve(ARCHIVE_URL, archive)
            archive_digest_ok(archive)
        extract_root = temp / "extracted"
        with zipfile.ZipFile(archive) as zip_handle:
            zip_handle.extractall(extract_root)
        yield find_course_root(extract_root)


def parse_resource_rows(contents: Path) -> dict[int, dict[str, Any]]:
    source = contents / "resource-index/index.htm"
    soup = BeautifulSoup(source.read_text(encoding="utf-8", errors="replace"), "html.parser")
    table = next(
        table
        for table in soup.select("table")
        if "LECTURE SLIDES" in normalized_text(table.get_text(" ")).upper()
    )
    rows: dict[int, dict[str, Any]] = {}
    for tr in table.select("tr"):
        cells = tr.find_all("td", recursive=False)
        if not cells:
            continue
        session_text = normalized_text(cells[0].get_text(" "))
        if not session_text.isdigit():
            continue
        session = int(session_text)
        links = tr.select("a[href]")
        courseware = next(
            a
            for a in links
            if "/contents/" in a["href"]
            and any(f"/contents/{unit}/" in a["href"] for unit in UNIT_INFO)
            and a["href"].endswith("index.htm")
        )
        reading = next(a for a in links if "/contents/readings/" in a["href"])
        in_class = next(a for a in links if "/contents/in-class-questions/" in a["href"])
        slides = [
            {
                "basename": Path(a["href"]).name,
                "label": re.sub(r"\s*\(PDF.*$", "", normalized_text(a.get_text(" "))),
            }
            for a in links
            if "/contents/lecture-slides/" in a["href"]
        ]
        code, title = normalized_text(courseware.get_text(" ")).split(" ", 1)
        unit_slug = next(unit for unit in UNIT_INFO if f"/contents/{unit}/" in courseware["href"])
        rows[session] = {
            "session": session,
            "unit_slug": unit_slug,
            "unit": UNIT_INFO[unit_slug][0],
            "unit_title": UNIT_INFO[unit_slug][1],
            "note_file": UNIT_INFO[unit_slug][2],
            "code": code,
            "title": title,
            "courseware_html": (source.parent / courseware["href"]).resolve(),
            "courseware_url": courseware.get("href"),
            "reading": Path(reading["href"]).name,
            "reading_label": normalized_text(reading.get_text(" ")),
            "in_class": Path(in_class["href"]).name,
            "slides": slides,
        }
    if len(rows) != EXPECTED["sessions"] or set(rows) != set(range(1, 36)):
        raise AssertionError(f"Expected 35 resource-index rows, found {len(rows)}")
    return rows


def prompt_data(assessment: Any) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for ordinal, question in enumerate(assessment.select("div.problem_question"), start=1):
        match = re.search(r"Q(\d+)", question.get("id", ""))
        prompt_number = int(match.group(1)) if match else ordinal
        prompt_id = f"Q{prompt_number}"

        question_clone = BeautifulSoup(str(question), "html.parser")
        fieldset = question_clone.select_one("fieldset")
        if fieldset:
            fieldset.decompose()
        question_text = normalized_text(question_clone.get_text(" "))

        options: list[str] = []
        correct: list[str] = []
        select = question.select_one("select")
        if select:
            response_type = "dropdown"
            for option in select.select("option"):
                text = normalized_text(option.get_text(" "))
                if text:
                    options.append(text)
                    if option.get("correct") == "true":
                        correct.append(text)
        elif question.select_one('input[type="checkbox"]'):
            response_type = "checkbox"
        elif question.select_one('input[type="radio"]'):
            response_type = "multiple_choice"
        elif question.select_one('input[id$="_tolerance"]'):
            response_type = "numeric"
        else:
            response_type = "text"

        for input_element in question.select('input[type="checkbox"], input[type="radio"]'):
            label = input_element.find_parent("label")
            choice = label.select_one("span.choice") if label else None
            text = normalized_text((choice or label).get_text(" ")) if (choice or label) else ""
            if text:
                options.append(text)
                if input_element.get("correct") == "true":
                    correct.append(text)

        hidden_answer = question.select_one(f'input[id="{prompt_id}_ans"]')
        if hidden_answer and hidden_answer.get("value"):
            correct.append(hidden_answer["value"])
        text_answer = question.select_one('input[type="text"][answer]')
        if text_answer and text_answer.get("answer"):
            correct.append(text_answer["answer"])
        answer_span = question.select_one(f'span[id="{prompt_id}_ans_span"]')
        if answer_span:
            value = normalized_text(answer_span.get_text(" "))
            value = re.sub(r"^Answer:\s*", "", value, flags=re.I)
            if value:
                correct.append(value)
        correct = list(dict.fromkeys(answer for answer in correct if answer))

        tolerance_element = question.select_one(f'input[id="{prompt_id}_tolerance"]')
        tolerance = tolerance_element.get("value", "") if tolerance_element else ""
        solution_element = assessment.select_one(f"#S{prompt_number}_div")
        solution = ""
        if solution_element:
            solution = markdownify(
                str(solution_element), heading_style="ATX", bullets="-"
            ).strip()
            solution = re.sub(r"\n{3,}", "\n\n", solution)

        if not correct and not solution:
            raise AssertionError(f"No answer or feedback for prompt {prompt_id}")
        prompts.append(
            {
                "prompt_id": prompt_id,
                "response_type": response_type,
                "question": question_text,
                "options": options,
                "correct_answers": correct,
                "tolerance": tolerance,
                "official_feedback": solution,
            }
        )
    return prompts


def parse_blocks(contents: Path, rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for unit_slug, (_unit_no, _unit_title, _note_file, offset) in UNIT_INFO.items():
        unit_index = contents / unit_slug / "index.htm"
        soup = BeautifulSoup(
            unit_index.read_text(encoding="utf-8", errors="replace"), "html.parser"
        )
        main = soup.select_one("main#course_inner_section")
        if main is None:
            raise AssertionError(f"No course content in {unit_index}")
        for link in main.select("a[href]"):
            label = normalized_text(link.get_text(" "))
            match = re.match(r"^(\d+)\.(\d+)\.(\d+)\s+(.+)$", label)
            if not match:
                continue
            unit_no, unit_session, block_order = map(int, match.group(1, 2, 3))
            session = offset + unit_session
            if unit_no != UNIT_INFO[unit_slug][0]:
                raise AssertionError(f"Block unit mismatch: {label}")
            html_path = (unit_index.parent / link["href"]).resolve()
            block_soup = BeautifulSoup(
                html_path.read_text(encoding="utf-8", errors="replace"), "html.parser"
            )
            assessment = block_soup.select_one("div.self_assessment")
            pdf_links = [
                (html_path.parent / a["href"]).resolve()
                for a in block_soup.select('a[href$=".pdf"]')
            ]
            transcript = next(
                (path for path in pdf_links if "lecture-slides" not in path.parts), None
            )
            slide_only_pdf = next(
                (path for path in pdf_links if "lecture-slides" in path.parts), None
            )
            if assessment is not None:
                block_type = "exercise"
            elif transcript is not None:
                block_type = "video"
            elif slide_only_pdf is not None:
                block_type = "slide_only"
            else:
                raise AssertionError(f"Cannot classify courseware block: {label}")

            title = match.group(4)
            number = f"{unit_no}.{unit_session}.{block_order}"
            canonical = block_soup.select_one('link[rel="canonical"]')
            official_url = canonical.get("href") if canonical else rows[session]["courseware_url"]
            block: dict[str, Any] = {
                "session": session,
                "unit": unit_no,
                "unit_title": UNIT_INFO[unit_slug][1],
                "number": number,
                "order": block_order,
                "title": title,
                "type": block_type,
                "optional": "optional" in title.lower(),
                "official_url": official_url,
                "source_html": html_path.relative_to(contents.parent).as_posix(),
                "_html_path": html_path,
            }
            if block_type == "video":
                srt_link = block_soup.select_one('a[href$=".srt"]')
                if srt_link is None:
                    raise AssertionError(f"Video has no SRT: {label}")
                srt_path = (html_path.parent / srt_link["href"]).resolve()
                page_text = html_path.read_text(encoding="utf-8", errors="replace")
                youtube_match = re.search(r"youtube\.com/v/([^'\"&,)]+)", page_text)
                archive_link = block_soup.select_one('a[href*="archive.org"][href$=".mp4"]')
                block.update(
                    {
                        "transcript_pdf": f"03_Video_Transcripts/{transcript.name}",
                        "caption_srt": f"04_Captions/{srt_path.name}",
                        "youtube_id": youtube_match.group(1) if youtube_match else "",
                        "youtube_url": (
                            f"https://www.youtube.com/watch?v={youtube_match.group(1)}"
                            if youtube_match
                            else ""
                        ),
                        "internet_archive_mp4_url": (
                            archive_link["href"] if archive_link else ""
                        ),
                        "_transcript_source": transcript,
                        "_caption_source": srt_path,
                    }
                )
            elif block_type == "exercise":
                prompts = prompt_data(assessment)
                exercise_name = (
                    f"S{session:02d}_{number}_{slugify(title)}.md"
                )
                block.update(
                    {
                        "exercise_markdown": f"08_Courseware_Exercises/{exercise_name}",
                        "prompt_count": len(prompts),
                        "prompts": prompts,
                    }
                )
            blocks.append(block)

    blocks.sort(key=lambda item: (item["session"], item["order"]))
    numbers = [block["number"] for block in blocks]
    if len(blocks) != EXPECTED["courseware_blocks"] or len(set(numbers)) != len(numbers):
        raise AssertionError(
            f"Expected 284 unique courseware blocks, found {len(blocks)}"
        )
    return blocks


def assign_slides(
    blocks: list[dict[str, Any]], rows: dict[int, dict[str, Any]]
) -> dict[int, list[dict[str, str]]]:
    supplemental: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_session: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        if block["type"] != "exercise":
            by_session[block["session"]].append(block)

    for session in range(1, 36):
        session_blocks = by_session[session]
        slides = rows[session]["slides"]
        if session == 1:
            if len(session_blocks) != 3 or len(slides) != 2:
                raise AssertionError("Unexpected Session 1 slide/video structure")
            mapping = [slides[0], slides[1], slides[1]]
        elif session == 2:
            if len(session_blocks) != 2 or len(slides) != 3:
                raise AssertionError("Unexpected Session 2 slide/video structure")
            mapping = slides[:2]
            supplemental[session].append(slides[2])
        else:
            if len(session_blocks) != len(slides):
                raise AssertionError(
                    f"Session {session}: {len(session_blocks)} non-exercise blocks but "
                    f"{len(slides)} slide decks"
                )
            mapping = slides
        for block, slide in zip(session_blocks, mapping):
            block["lecture_slide"] = f"02_Lecture_Slides/{slide['basename']}"
    return supplemental


def image_sources(contents: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".gif"}
    images = [
        path
        for path in contents.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and not path.name.lower().startswith("6-042js15")
    ]
    images.sort(key=lambda path: path.relative_to(contents).as_posix().lower())
    if len(images) != EXPECTED["courseware_images"]:
        raise AssertionError(f"Expected 17 courseware images, found {len(images)}")
    return images


def rewrite_exercise_markdown(
    block: dict[str, Any],
    contents: Path,
    image_map: dict[Path, str],
) -> str:
    html_path: Path = block["_html_path"]
    soup = BeautifulSoup(
        html_path.read_text(encoding="utf-8", errors="replace"), "html.parser"
    )
    original = soup.select_one("div.self_assessment")
    if original is None:
        raise AssertionError(f"Exercise missing self_assessment: {block['number']}")
    assessment = BeautifulSoup(str(original), "html.parser")

    for image in assessment.select("img[src]"):
        source = (html_path.parent / image["src"]).resolve()
        if source not in image_map:
            raise AssertionError(f"Unmapped courseware image: {source}")
        image["src"] = "../" + image_map[source]

    for select in assessment.select("select"):
        options = [
            normalized_text(option.get_text(" "))
            for option in select.select("option")
            if normalized_text(option.get_text(" "))
        ]
        select.replace_with("[Choose one: " + " | ".join(options) + "]")
    for input_element in assessment.select('input[type="checkbox"]'):
        input_element.replace_with("[ ] ")
    for input_element in assessment.select('input[type="radio"]'):
        input_element.replace_with("( ) ")
    for input_element in assessment.select('input[type="text"]'):
        input_element.replace_with("[response]")
    for selector in (
        'input[type="hidden"]',
        "span.visually-hidden",
        "span.nostatus",
        "p.nostatus",
        'span[id$="_ans_span"]',
        "p.problem_answer",
        "div.problem_solution",
        "div.action",
        "button",
        "script",
        "style",
    ):
        for element in assessment.select(selector):
            element.decompose()

    body = markdownify(str(assessment), heading_style="ATX", bullets="-").strip()
    body = re.sub(r"\n{3,}", "\n\n", body)

    def replace_feedback_image(match: re.Match[str]) -> str:
        alt, raw_target = match.group(1), match.group(2)
        target = raw_target.strip().strip("<>")
        if "://" in target:
            return match.group(0)
        source = (html_path.parent / target).resolve()
        if source not in image_map:
            raise AssertionError(f"Unmapped image in official feedback: {source}")
        return f"![{alt}](../{image_map[source]})"

    for prompt in block["prompts"]:
        prompt["official_feedback"] = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            replace_feedback_image,
            prompt["official_feedback"],
        )

    lines = [
        "---",
        "aliases: []",
        "tags:",
        "  - course/MIT-6.042J",
        "  - source/official-courseware",
        "type: official-exercise",
        f"session: {block['session']}",
        f'block: "{block["number"]}"',
        "official_answer: true",
        "---",
        f"# {block['number']} {block['title']}",
        "",
        "> [!info] 来源说明",
        "> 题目、正确答案与反馈均从 MIT OCW 官方离线课程包提取；动态作答控件已转换为静态 Markdown。",
        f"> [官方课程页]({block['official_url']})",
        "",
        "## Official exercise",
        "",
        body,
        "",
        "## Official answers and feedback",
        "",
    ]
    for prompt in block["prompts"]:
        lines.append(f"> [!answer]- {prompt['prompt_id']} — official answer")
        answers = prompt["correct_answers"]
        if answers:
            lines.append("> **Answer:** " + "; ".join(answers))
        else:
            lines.append("> **Answer:** See the official feedback below.")
        if prompt["tolerance"]:
            lines.append(f"> **Tolerance:** {prompt['tolerance']}")
        if prompt["official_feedback"]:
            lines.append(">")
            lines.append("> **Official feedback:**")
            for solution_line in prompt["official_feedback"].splitlines():
                lines.append("> " + solution_line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def pdf_pages(path: Path) -> int:
    return len(PdfReader(str(path), strict=False).pages)


def file_record(
    *,
    source: Path,
    destination: Path,
    output: Path,
    contents: Path,
    asset_id: str,
    category: str,
    session: str | int = "",
    unit: str | int = "",
    block: str = "",
    title: str = "",
    official_url: str = "",
    provenance: str = "Official MIT OCW Spring 2015 course package",
    year: str | int = 2015,
) -> dict[str, Any]:
    relative_source = source.relative_to(contents.parent).as_posix()
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash and source.suffix.lower() != ".htm":
        raise AssertionError(f"Copied asset checksum mismatch: {source} -> {destination}")
    return {
        "asset_id": asset_id,
        "category": category,
        "session": session,
        "unit": unit,
        "block": block,
        "resource_title": title,
        "original_basename": source.name,
        "local_path": destination.relative_to(output).as_posix(),
        "source_relative_path": relative_source,
        "official_url": official_url
        or official_asset_url(source.relative_to(contents)),
        "provenance": provenance,
        "year": year,
        "file_format": destination.suffix.lstrip(".").lower(),
        "pages": pdf_pages(destination) if destination.suffix.lower() == ".pdf" else "",
        "bytes": destination.stat().st_size,
        "sha256": destination_hash,
    }


def copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def problem_numbers_and_pages(path: Path) -> list[tuple[int, int]]:
    page_texts = [page.extract_text() or "" for page in PdfReader(str(path), strict=False).pages]
    all_numbers: set[int] = set()
    for text in page_texts:
        all_numbers.update(
            int(value) for value in re.findall(r"Problem\s+(\d+)\b", text)
        )
    maximum = 0
    while maximum + 1 in all_numbers:
        maximum += 1
    result: list[tuple[int, int]] = []
    for number in range(1, maximum + 1):
        pattern = re.compile(rf"Problem\s+{number}\b")
        page = next(
            (index for index, text in enumerate(page_texts, start=1) if pattern.search(text)),
            1,
        )
        result.append((number, page))
    return result


def exercise_image_usage(
    blocks: list[dict[str, Any]], image_map: dict[Path, str]
) -> dict[Path, list[dict[str, Any]]]:
    usage: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        if block["type"] != "exercise":
            continue
        html_path: Path = block["_html_path"]
        soup = BeautifulSoup(
            html_path.read_text(encoding="utf-8", errors="replace"), "html.parser"
        )
        assessment = soup.select_one("div.self_assessment")
        if assessment is None:
            continue
        images: list[str] = []
        for image in assessment.select("img[src]"):
            source = (html_path.parent / image["src"]).resolve()
            if source in image_map:
                usage[source].append(block)
                images.append(image_map[source])
        if images:
            block["images"] = list(dict.fromkeys(images))
    return usage


def build_courseware_json(
    blocks: list[dict[str, Any]],
    rows: dict[int, dict[str, Any]],
    supplemental: dict[int, list[dict[str, str]]],
) -> dict[str, Any]:
    sessions = []
    for session in range(1, 36):
        session_blocks = [block for block in blocks if block["session"] == session]
        clean_blocks = []
        for block in session_blocks:
            clean_blocks.append(
                {
                    key: value
                    for key, value in block.items()
                    if not key.startswith("_") and key not in {"unit_title"}
                }
            )
        sessions.append(
            {
                "session": session,
                "unit": rows[session]["unit"],
                "unit_title": rows[session]["unit_title"],
                "courseware_code": rows[session]["code"],
                "title": rows[session]["title"],
                "reading_pdf": f"01_Session_Readings/{rows[session]['reading']}",
                "reading_assignment": rows[session]["reading_label"],
                "in_class_pdf": f"05_In_Class_Questions/{rows[session]['in_class']}",
                "supplemental_slides": [
                    f"02_Lecture_Slides/{slide['basename']}"
                    for slide in supplemental.get(session, [])
                ],
                "blocks": clean_blocks,
            }
        )
    counts = Counter(block["type"] for block in blocks)
    prompt_count = sum(block.get("prompt_count", 0) for block in blocks)
    return {
        "course": COURSE_ID,
        "official_course_url": COURSE_URL,
        "source_archive_url": ARCHIVE_URL,
        "source_archive_sha256": ARCHIVE_SHA256,
        "counts": {
            "sessions": len(sessions),
            "blocks": len(blocks),
            "video": counts["video"],
            "exercise": counts["exercise"],
            "slide_only": counts["slide_only"],
            "prompts": prompt_count,
        },
        "sessions": sessions,
    }


def build_index(
    courseware: dict[str, Any], rows: dict[int, dict[str, Any]]
) -> str:
    lines = [
        "---",
        "aliases:",
        "  - MIT 6.042J materials index",
        "  - Mathematics for Computer Science materials",
        "tags:",
        "  - course/MIT-6.042J",
        "  - type/resource-index",
        "---",
        "# MIT 6.042J official materials index",
        "",
        f"课程来源：[MIT OCW 6.042J Spring 2015]({COURSE_URL})。本目录保存官方课程包中有直接学习价值的材料；视频不在本地重复保存，使用官方 transcript、SRT 及在线链接。",
        "",
        "## Inventory",
        "",
        "| Material | Count | Location |",
        "|---|---:|---|",
        "| Session readings | 35 | `01_Session_Readings/` |",
        "| Lecture slides | 113 | `02_Lecture_Slides/` |",
        "| Video transcripts | 111 | `03_Video_Transcripts/` |",
        "| Captions | 111 | `04_Captions/` |",
        "| In-class question sets | 35 | `05_In_Class_Questions/` |",
        "| Problem sets | 12 | `06_Problem_Sets/` |",
        "| Exams | 4 | `07_Exams/` |",
        "| Extracted official exercise blocks | 171 / 376 prompts | `08_Courseware_Exercises/` |",
        "| Courseware figures | 17 | `09_Courseware_Images/` |",
        "| Spring 2015 textbook | 1 | [[99_Books/MIT6_042JS15_textbook.pdf|open PDF]] |",
        "",
        "> [!note] Slide provenance",
        "> 113 份 slides 全部来自官方 Spring 2015 课程包。其中 63 份文件名为 `MIT6_042JS15_*`；50 份为官方包中随附的 `MIT6_042JS16_*` 后续/替换版本。索引保留原年份，不把 S16 文件误标成 S15 原稿。",
        "",
        "> [!info] Metadata",
        "> `manifest.csv` 保存逐文件来源、页数、大小和 SHA-256；`courseware_blocks.json` 保存 284 个 block 的官方顺序与答案数据；`problem_coverage.csv` 是 596 个官方问题或 prompt 的题解覆盖台账。",
        "",
    ]
    previous_unit = None
    for session in courseware["sessions"]:
        if session["unit"] != previous_unit:
            lines.extend([f"## {session['unit_title']}", ""])
            previous_unit = session["unit"]
        session_number = session["session"]
        reading = Path(session["reading_pdf"]).name
        in_class = Path(session["in_class_pdf"]).name
        blocks = session["blocks"]
        video_count = sum(block["type"] == "video" for block in blocks)
        exercise_count = sum(block["type"] == "exercise" for block in blocks)
        lines.extend(
            [
                f"### Session {session_number}: {session['title']}",
                "",
                f"- Reading: [[01_Session_Readings/{reading}|{session['reading_assignment']}]]",
                f"- In-class questions: [[05_In_Class_Questions/{in_class}|Session {session_number} PDF]]",
                f"- Courseware: {len(blocks)} blocks ({video_count} videos, {exercise_count} exercise blocks)",
                "",
            ]
        )
        for block in blocks:
            links = []
            if block.get("lecture_slide"):
                name = Path(block["lecture_slide"]).name
                links.append(f"[[02_Lecture_Slides/{name}|slides]]")
            if block["type"] == "video":
                transcript = Path(block["transcript_pdf"]).name
                caption = Path(block["caption_srt"]).name
                links.extend(
                    [
                        f"[[03_Video_Transcripts/{transcript}|transcript]]",
                        f"[[04_Captions/{caption}|SRT]]",
                    ]
                )
                if block.get("youtube_url"):
                    links.append(f"[video]({block['youtube_url']})")
            elif block["type"] == "exercise":
                exercise = Path(block["exercise_markdown"]).name
                links.append(
                    f"[[08_Courseware_Exercises/{exercise}|exercise + official answer]]"
                )
            lines.append(
                f"- **{block['number']} {block['title']}** — " + ", ".join(links)
            )
        for slide in session["supplemental_slides"]:
            name = Path(slide).name
            lines.append(f"- **Supplemental slide deck** — [[02_Lecture_Slides/{name}|{name}]]")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def coverage_rows(
    blocks: list[dict[str, Any]],
    output: Path,
    rows: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for block in blocks:
        if block["type"] != "exercise":
            continue
        note_file = rows[block["session"]]["note_file"]
        for prompt in block["prompts"]:
            prompt_id = prompt["prompt_id"]
            coverage.append(
                {
                    "coverage_id": f"online-S{block['session']:02d}-{block['number']}-{prompt_id}",
                    "category": "online_feedback",
                    "session": block["session"],
                    "unit": block["unit"],
                    "source_file": block["exercise_markdown"],
                    "source_page": "",
                    "problem_number": "",
                    "prompt_id": prompt_id,
                    "official_answer_available": "yes",
                    "note_file": note_file,
                    "note_anchor": (
                        f"#session-{block['session']}-online-{block['number'].replace('.', '-')}-"
                        f"{prompt_id.lower()}"
                    ),
                    "solution_status": "official answer archived; note integration pending",
                    "verification_method": "official OCW HTML correct-answer and feedback fields",
                }
            )

    for pset, (start_session, end_session) in PSET_SESSION_RANGES.items():
        source = output / "06_Problem_Sets" / f"MIT6_042JS15_ps{pset}.pdf"
        unit, _unit_title, note_file = unit_for_session(end_session)
        for problem, page in problem_numbers_and_pages(source):
            coverage.append(
                {
                    "coverage_id": f"ps{pset}-p{problem}",
                    "category": "problem_set",
                    "session": f"{start_session}-{end_session}",
                    "unit": unit,
                    "source_file": f"06_Problem_Sets/{source.name}",
                    "source_page": page,
                    "problem_number": problem,
                    "prompt_id": "",
                    "official_answer_available": "no",
                    "note_file": note_file,
                    "note_anchor": f"#problem-set-{pset}-problem-{problem}",
                    "solution_status": "non-official solution pending",
                    "verification_method": "independent derivation and small-case checks required",
                }
            )

    for session in range(1, 36):
        source = output / "05_In_Class_Questions" / f"MIT6_042JS15_cp{session}.pdf"
        unit, _unit_title, note_file = unit_for_session(session)
        for problem, page in problem_numbers_and_pages(source):
            coverage.append(
                {
                    "coverage_id": f"class-S{session:02d}-p{problem}",
                    "category": "in_class",
                    "session": session,
                    "unit": unit,
                    "source_file": f"05_In_Class_Questions/{source.name}",
                    "source_page": page,
                    "problem_number": problem,
                    "prompt_id": "",
                    "official_answer_available": "no",
                    "note_file": note_file,
                    "note_anchor": f"#session-{session}-in-class-problem-{problem}",
                    "solution_status": "non-official solution pending",
                    "verification_method": "independent derivation and small-case checks required",
                }
            )

    for filename, (exam_title, checkpoint, note_file) in EXAM_INFO.items():
        source = output / "07_Exams" / filename
        unit = unit_for_session(checkpoint)[0]
        exam_slug = exam_title.lower().replace(" ", "-")
        for problem, page in problem_numbers_and_pages(source):
            coverage.append(
                {
                    "coverage_id": f"{exam_slug}-p{problem}",
                    "category": "exam",
                    "session": checkpoint,
                    "unit": unit,
                    "source_file": f"07_Exams/{filename}",
                    "source_page": page,
                    "problem_number": problem,
                    "prompt_id": "",
                    "official_answer_available": "no",
                    "note_file": note_file,
                    "note_anchor": f"#{exam_slug}-problem-{problem}",
                    "solution_status": "non-official solution pending",
                    "verification_method": "independent derivation and full exam consistency check required",
                }
            )
    return coverage


def integrate_completed_notes(
    coverage: list[dict[str, Any]], course_dir: Path
) -> list[dict[str, Any]]:
    """Point coverage rows at real note headings when the finished notes exist.

    The material synchronizer is also useful before the notes have been written,
    so missing headings deliberately leave the original ``pending`` fields
    untouched.  Once the five main notes are present, a later synchronization is
    therefore reproducible without erasing the completed coverage ledger.
    """

    heading_index: dict[str, list[str]] = {}
    for note_file in {str(row["note_file"]) for row in coverage}:
        path = course_dir / note_file
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Coverage points at the stable top-level course sections, never at a
        # similarly named Session self-check or error-diagnosis subsection.
        heading_index[note_file] = re.findall(r"^##\s+(.+?)\s*$", text, re.M)

    def exact_heading(note_file: str, pattern: str) -> str | None:
        regex = re.compile(pattern, re.I)
        matches = [heading for heading in heading_index.get(note_file, []) if regex.search(heading)]
        return matches[0] if len(matches) == 1 else None

    for row in coverage:
        note_file = str(row["note_file"])
        category = str(row["category"])
        heading: str | None = None
        if category in {"online_feedback", "in_class"}:
            session = int(row["session"])
            heading = exact_heading(note_file, rf"^Session {session}\b")
        elif category == "problem_set":
            match = re.match(r"ps(\d+)-", str(row["coverage_id"]))
            if match:
                pset = int(match.group(1))
                heading = exact_heading(note_file, rf"^Problem Set {pset}\b")
        elif category == "exam":
            if str(row["coverage_id"]).startswith("final-exam-"):
                heading = exact_heading(note_file, r"Final Exam.*题解")
            else:
                match = re.match(r"midterm-(\d+)-", str(row["coverage_id"]))
                if match:
                    midterm = int(match.group(1))
                    heading = exact_heading(note_file, rf"^Midterm {midterm}\b")

        if heading is None:
            continue
        row["note_anchor"] = f"#{heading}"
        if category == "online_feedback":
            row["solution_status"] = (
                "integrated; official answer and feedback explained in course note"
            )
            row["verification_method"] = (
                "official OCW correct-answer and feedback fields; checked in course note"
            )
        elif category == "exam":
            row["solution_status"] = (
                "integrated; complete independent solution in course note"
            )
            row["verification_method"] = (
                "independent derivation; full-exam consistency check in course note"
            )
        else:
            row["solution_status"] = (
                "integrated; complete independent solution in course note"
            )
            row["verification_method"] = (
                "independent derivation; boundary and small-case checks in course note"
            )
    return coverage


def write_csv(path: Path, fieldnames: tuple[str, ...], records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def synchronize(source_root: Path, output: Path) -> None:
    contents = source_root / "contents"
    rows = parse_resource_rows(contents)
    blocks = parse_blocks(contents, rows)
    supplemental = assign_slides(blocks, rows)

    output.mkdir(parents=True, exist_ok=True)
    for directory in OUTPUT_DIRS:
        (output / directory).mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []

    # Session readings and the Spring 2015 textbook.
    for source in sorted((contents / "readings").glob("*.pdf")):
        if source.name == "MIT6_042JS15_textbook.pdf":
            category = "book"
            destination = output / "99_Books" / source.name
            session: str | int = ""
            unit: str | int = ""
            title = "Mathematics for Computer Science (Spring 2015 textbook)"
        else:
            match = re.search(r"Session(\d+)", source.name)
            if not match:
                raise AssertionError(f"Unrecognized reading filename: {source.name}")
            session = int(match.group(1))
            unit = rows[session]["unit"]
            category = "session_reading"
            destination = output / "01_Session_Readings" / source.name
            title = f"Session {session}: {rows[session]['reading_label']}"
        copy_asset(source, destination)
        manifests.append(
            file_record(
                source=source,
                destination=destination,
                output=output,
                contents=contents,
                asset_id=f"{category}-{source.stem}",
                category=category,
                session=session,
                unit=unit,
                title=title,
            )
        )

    # Lecture slides. Session ownership comes from the official resource table.
    slide_session: dict[str, int] = {}
    slide_label: dict[str, str] = {}
    for session, row in rows.items():
        for slide in row["slides"]:
            slide_session[slide["basename"]] = session
            slide_label[slide["basename"]] = slide["label"]
    slide_usage: dict[str, list[str]] = defaultdict(list)
    for block in blocks:
        if block.get("lecture_slide"):
            slide_usage[Path(block["lecture_slide"]).name].append(block["number"])
    for session, slides in supplemental.items():
        for slide in slides:
            slide_usage[slide["basename"]].append(f"S{session}-supplemental")

    for source in sorted((contents / "lecture-slides").glob("*.pdf")):
        destination = output / "02_Lecture_Slides" / source.name
        copy_asset(source, destination)
        session = slide_session[source.name]
        is_s16 = source.name.startswith("MIT6_042JS16_")
        provenance = (
            "Official later/replacement 2016 slide deck bundled with the Spring 2015 OCW package"
            if is_s16
            else "Official Spring 2015 slide deck"
        )
        manifests.append(
            file_record(
                source=source,
                destination=destination,
                output=output,
                contents=contents,
                asset_id=f"slide-{source.stem}",
                category="lecture_slide",
                session=session,
                unit=rows[session]["unit"],
                block=";".join(slide_usage[source.name]),
                title=slide_label[source.name],
                provenance=provenance,
                year=2016 if is_s16 else 2015,
            )
        )

    # Transcripts and captions follow the 111 video blocks.
    for block in blocks:
        if block["type"] != "video":
            continue
        transcript_source: Path = block["_transcript_source"]
        transcript_destination = output / block["transcript_pdf"]
        copy_asset(transcript_source, transcript_destination)
        manifests.append(
            file_record(
                source=transcript_source,
                destination=transcript_destination,
                output=output,
                contents=contents,
                asset_id=f"transcript-{transcript_source.stem}",
                category="video_transcript",
                session=block["session"],
                unit=block["unit"],
                block=block["number"],
                title=block["title"],
                official_url=block["official_url"],
            )
        )
        caption_source: Path = block["_caption_source"]
        caption_destination = output / block["caption_srt"]
        copy_asset(caption_source, caption_destination)
        manifests.append(
            file_record(
                source=caption_source,
                destination=caption_destination,
                output=output,
                contents=contents,
                asset_id=f"caption-{caption_source.stem}",
                category="caption",
                session=block["session"],
                unit=block["unit"],
                block=block["number"],
                title=block["title"],
                official_url=block["official_url"],
            )
        )

    # In-class questions.
    for session in range(1, 36):
        source = contents / "in-class-questions" / rows[session]["in_class"]
        destination = output / "05_In_Class_Questions" / source.name
        copy_asset(source, destination)
        manifests.append(
            file_record(
                source=source,
                destination=destination,
                output=output,
                contents=contents,
                asset_id=f"in-class-S{session:02d}",
                category="in_class_questions",
                session=session,
                unit=rows[session]["unit"],
                title=f"Session {session} In-Class Questions",
            )
        )

    # Problem sets.
    for pset, (start_session, end_session) in PSET_SESSION_RANGES.items():
        source = contents / "assignments" / f"MIT6_042JS15_ps{pset}.pdf"
        destination = output / "06_Problem_Sets" / source.name
        copy_asset(source, destination)
        unit = unit_for_session(end_session)[0]
        manifests.append(
            file_record(
                source=source,
                destination=destination,
                output=output,
                contents=contents,
                asset_id=f"problem-set-{pset}",
                category="problem_set",
                session=f"{start_session}-{end_session}",
                unit=unit,
                title=f"Problem Set {pset}",
            )
        )

    # Exams.
    for filename, (title, checkpoint, _note_file) in EXAM_INFO.items():
        source = contents / "exams" / filename
        destination = output / "07_Exams" / filename
        copy_asset(source, destination)
        manifests.append(
            file_record(
                source=source,
                destination=destination,
                output=output,
                contents=contents,
                asset_id=title.lower().replace(" ", "-"),
                category="exam",
                session=checkpoint,
                unit=unit_for_session(checkpoint)[0],
                title=title,
            )
        )

    # Static figures used inside the official interactive exercises.
    image_map: dict[Path, str] = {}
    for source in image_sources(contents):
        relative = source.relative_to(contents)
        local = Path("09_Courseware_Images") / relative
        image_map[source.resolve()] = local.as_posix()
        copy_asset(source, output / local)
    usage = exercise_image_usage(blocks, image_map)
    for source, local in image_map.items():
        users = usage.get(source, [])
        session = users[0]["session"] if users else ""
        unit = users[0]["unit"] if users else ""
        block_numbers = ";".join(user["number"] for user in users)
        manifests.append(
            file_record(
                source=source,
                destination=output / local,
                output=output,
                contents=contents,
                # The same bitmap legitimately appears in more than one
                # exercise path, so a content hash alone is not a row ID.
                asset_id=(
                    f"courseware-image-{sha256(source)[:12]}-"
                    f"{hashlib.sha256(local.as_posix().encode('utf-8')).hexdigest()[:8]}"
                ),
                category="courseware_image",
                session=session,
                unit=unit,
                block=block_numbers,
                title=source.stem.replace("_", " "),
            )
        )

    # Static Markdown versions of all 171 official interactive exercise blocks.
    for block in blocks:
        if block["type"] != "exercise":
            continue
        destination = output / block["exercise_markdown"]
        destination.write_text(
            rewrite_exercise_markdown(block, contents, image_map), encoding="utf-8"
        )
        source: Path = block["_html_path"]
        manifests.append(
            file_record(
                source=source,
                destination=destination,
                output=output,
                contents=contents,
                asset_id=f"exercise-{block['number']}",
                category="courseware_exercise",
                session=block["session"],
                unit=block["unit"],
                block=block["number"],
                title=block["title"],
                official_url=block["official_url"],
                provenance=(
                    "Official interactive exercise extracted from OCW HTML; "
                    "correct-answer and feedback fields preserved"
                ),
            )
        )

    courseware = build_courseware_json(blocks, rows, supplemental)
    (output / "courseware_blocks.json").write_text(
        json.dumps(courseware, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "index.md").write_text(build_index(courseware, rows), encoding="utf-8")
    manifests.sort(key=lambda row: (row["category"], str(row["session"]), row["local_path"]))
    write_csv(output / "manifest.csv", MANIFEST_FIELDS, manifests)
    coverage = integrate_completed_notes(coverage_rows(blocks, output, rows), output.parent)
    write_csv(output / "problem_coverage.csv", COVERAGE_FIELDS, coverage)
    verify_output(output)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_output(output: Path) -> None:
    # The manifest is the authoritative synchronization boundary.  In
    # particular, iCloud may create conflict copies such as ``name 2.pdf`` in
    # the same directory; those files are not course assets produced by this
    # script and must not change verification counts.
    manifest = read_csv(output / "manifest.csv")
    asset_ids = [row["asset_id"] for row in manifest]
    if len(asset_ids) != len(set(asset_ids)):
        raise AssertionError("manifest.csv contains duplicate asset IDs")
    local_paths = [row["local_path"] for row in manifest]
    if len(local_paths) != len(set(local_paths)):
        raise AssertionError("manifest.csv contains duplicate local paths")

    for row in manifest:
        path = output / row["local_path"]
        if not path.is_file():
            raise AssertionError(f"Manifest target missing: {path}")
        if sha256(path) != row["sha256"]:
            raise AssertionError(f"Manifest checksum mismatch: {path}")

    def assets(
        *, category: str | None = None, file_format: str | None = None
    ) -> list[Path]:
        return [
            output / row["local_path"]
            for row in manifest
            if (category is None or row["category"] == category)
            and (file_format is None or row["file_format"] == file_format)
        ]

    pdfs = assets(file_format="pdf")
    srts = assets(category="caption", file_format="srt")
    images = assets(category="courseware_image")
    exercise_notes = assets(category="courseware_exercise", file_format="md")
    slides = assets(category="lecture_slide", file_format="pdf")
    counts = {
        "pdf": len(pdfs),
        "readings": len(assets(category="session_reading", file_format="pdf")),
        "books": len(assets(category="book", file_format="pdf")),
        "slides": len(slides),
        "slides_s15": sum(path.name.startswith("MIT6_042JS15_") for path in slides),
        "slides_s16": sum(path.name.startswith("MIT6_042JS16_") for path in slides),
        "transcripts": len(assets(category="video_transcript", file_format="pdf")),
        "captions": len(srts),
        "in_class": len(assets(category="in_class_questions", file_format="pdf")),
        "problem_sets": len(assets(category="problem_set", file_format="pdf")),
        "exams": len(assets(category="exam", file_format="pdf")),
        "courseware_images": len(images),
        "exercise_blocks": len(exercise_notes),
    }
    for key, actual in counts.items():
        expected = EXPECTED[key]
        if actual != expected:
            raise AssertionError(f"{key}: expected {expected}, found {actual}")

    for pdf in pdfs:
        if pdf.read_bytes()[:4] != b"%PDF" or pdf_pages(pdf) < 1:
            raise AssertionError(f"Invalid PDF: {pdf}")
    for caption in srts:
        if "-->" not in caption.read_text(encoding="utf-8", errors="replace"):
            raise AssertionError(f"Invalid SRT: {caption}")

    forbidden = []
    for suffix in ("*.mp4", "*.zip", "*.js", "*.xml", "*.html", "*.htm"):
        forbidden.extend(output.rglob(suffix))
    if forbidden:
        raise AssertionError(f"Forbidden offline-runtime/media files found: {forbidden[:5]}")

    courseware = json.loads((output / "courseware_blocks.json").read_text(encoding="utf-8"))
    json_counts = courseware["counts"]
    expected_json = {
        "sessions": EXPECTED["sessions"],
        "blocks": EXPECTED["courseware_blocks"],
        "video": EXPECTED["video_blocks"],
        "exercise": EXPECTED["exercise_blocks"],
        "slide_only": EXPECTED["slide_only_blocks"],
        "prompts": EXPECTED["prompts"],
    }
    if json_counts != expected_json:
        raise AssertionError(f"Courseware counts differ: {json_counts}")

    coverage = read_csv(output / "problem_coverage.csv")
    categories = Counter(row["category"] for row in coverage)
    expected_coverage = {
        "online_feedback": EXPECTED["coverage_online"],
        "problem_set": EXPECTED["coverage_problem_sets"],
        "in_class": EXPECTED["coverage_in_class"],
        "exam": EXPECTED["coverage_exams"],
    }
    if len(coverage) != EXPECTED["coverage_rows"] or categories != expected_coverage:
        raise AssertionError(
            f"Coverage mismatch: {len(coverage)} rows, categories {dict(categories)}"
        )
    coverage_ids = [row["coverage_id"] for row in coverage]
    if len(coverage_ids) != len(set(coverage_ids)):
        raise AssertionError("problem_coverage.csv contains duplicate coverage IDs")

    print("MIT 6.042J materials verified")
    print(json.dumps({**counts, **expected_json, "coverage_rows": len(coverage)}, indent=2))


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "MIT_OCW_6.042J_Materials"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Extracted course archive root")
    parser.add_argument("--archive", type=Path, help="Official course ZIP (verified before use)")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--verify-only", action="store_true", help="Validate an existing synchronized directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.verify_only:
        verify_output(output)
        return
    with course_source(args) as source_root:
        synchronize(source_root, output)


if __name__ == "__main__":
    main()
