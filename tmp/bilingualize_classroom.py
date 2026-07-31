#!/usr/bin/env python3
"""Generate an apply_patch payload that only inserts bilingual English layers."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from decimal import Decimal
from pathlib import Path


CJK_RE = re.compile(r"[\u3400-\u9fff]")
CHINESE_PUNCT_RE = re.compile(r"[，。；：、！？（）《》【】]")
START = "<!-- bilingual-en:start -->"
END = "<!-- bilingual-en:end -->"
WIKILINK_RE = re.compile(r"!?\[\[([^\]]+)\]\]")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
QUOTE_RE = re.compile(r"^(\s*(?:>\s*)+)(.*)$")
FENCE_RE = re.compile(r"^(?:```|~~~)")
PLACEHOLDER_WORDS = [
    "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF",
    "HOTEL", "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER",
    "OSCAR", "PAPA", "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM",
    "VICTOR", "WHISKEY", "XRAY",
]

# Reviewed terminology anchors.  They protect high-value academic terms from
# literal machine translation while the surrounding sentence is translated.
TERM_GLOSSARY = [
    ("资本资产定价模型", "capital asset pricing model"),
    ("套利定价理论", "arbitrage pricing theory"),
    ("消费资本资产定价模型", "consumption-based capital asset pricing model"),
    ("资产组合理论", "portfolio theory"),
    ("现代投资组合理论", "modern portfolio theory"),
    ("均值—方差", "mean–variance"),
    ("均值-方差", "mean–variance"),
    ("有效市场假说", "efficient market hypothesis"),
    ("弱式有效市场", "weak-form efficient market"),
    ("半强式有效市场", "semi-strong-form efficient market"),
    ("强式有效市场", "strong-form efficient market"),
    ("有效前沿", "efficient frontier"),
    ("系统性风险", "systematic risk"),
    ("非系统性风险", "idiosyncratic risk"),
    ("无风险利率", "risk-free rate"),
    ("无风险收益", "risk-free return"),
    ("风险溢价", "risk premium"),
    ("证券投资学", "Securities Investment"),
    ("证券市场", "securities market"),
    ("一级市场", "primary market"),
    ("二级市场", "secondary market"),
    ("集合竞价", "call auction"),
    ("连续竞价", "continuous auction"),
    ("价格发现", "price discovery"),
    ("沪深300指数", "CSI 300 Index"),
    ("技术分析", "technical analysis"),
    ("基本面分析", "fundamental analysis"),
    ("移动平均", "moving average"),
    ("支撑与阻力", "support and resistance"),
    ("市盈率", "price-to-earnings ratio"),
    ("市净率", "price-to-book ratio"),
    ("市销率", "price-to-sales ratio"),
    ("现金流量表", "cash-flow statement"),
    ("资产负债表", "balance sheet"),
    ("利润表", "income statement"),
    ("杜邦分析", "DuPont analysis"),
    ("财务报表分析", "financial-statement analysis"),
    ("自由现金流", "free cash flow"),
    ("加权平均资本成本", "weighted average cost of capital"),
    ("净现值", "net present value"),
    ("内部收益率", "internal rate of return"),
    ("资本预算", "capital budgeting"),
    ("货币政策", "monetary policy"),
    ("中央银行", "central bank"),
    ("联邦储备体系", "Federal Reserve System"),
    ("货币供给", "money supply"),
    ("基础货币", "monetary base"),
    ("货币乘数", "money multiplier"),
    ("公开市场操作", "open-market operations"),
    ("存款准备金率", "reserve requirement ratio"),
    ("贴现率", "discount rate"),
    ("逆向选择", "adverse selection"),
    ("道德风险", "moral hazard"),
    ("信息不对称", "information asymmetry"),
    ("利率期限结构", "term structure of interest rates"),
    ("收益率曲线", "yield curve"),
    ("流动性偏好理论", "liquidity-preference theory"),
    ("购买力平价", "purchasing-power parity"),
    ("外汇市场", "foreign-exchange market"),
    ("价值-at-风险", "value at risk"),
    ("风险价值", "value at risk"),
    ("在险价值", "value at risk"),
    ("历史模拟法", "historical simulation"),
    ("蒙特卡洛模拟", "Monte Carlo simulation"),
    ("极值理论", "extreme-value theory"),
    ("压力测试", "stress testing"),
    ("情景分析", "scenario analysis"),
    ("利率风险", "interest-rate risk"),
    ("违约风险", "default risk"),
    ("对手信用风险", "counterparty credit risk"),
    ("信用风险", "credit risk"),
    ("操作风险", "operational risk"),
    ("市场风险", "market risk"),
    ("风险暴露", "risk exposure"),
    ("巴塞尔协议", "Basel Accord"),
    ("偿付能力法案II", "Solvency II"),
    ("场外衍生产品", "over-the-counter derivatives"),
    ("波动率", "volatility"),
    ("相关系数", "correlation coefficient"),
    ("人口转型", "demographic transition"),
    ("刘易斯二元经济模型", "Lewis dual-sector model"),
    ("哈罗德—多马模型", "Harrod–Domar model"),
    ("哈罗德-多马模型", "Harrod–Domar model"),
    ("索洛增长模型", "Solow growth model"),
    ("内生增长理论", "endogenous growth theory"),
    ("经济增长", "economic growth"),
    ("经济发展", "economic development"),
]


def strip_quote(line: str) -> str:
    match = QUOTE_RE.match(line)
    return match.group(2) if match else line


def structural_core(line: str) -> str:
    """Remove quote/list indentation only for detecting fences and math."""
    line = strip_quote(line).lstrip()
    line = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line)
    return line.strip()


def visible_text(text: str) -> str:
    """Approximate Obsidian reading-view text for the no-CJK check."""

    def wiki(match: re.Match[str]) -> str:
        spec = match.group(1)
        if match.group(0).startswith("!"):
            return ""
        if "|" in spec:
            return spec.split("|", 1)[1]
        return spec.split("#", 1)[0]

    text = WIKILINK_RE.sub(wiki, text)
    text = MD_LINK_RE.sub(lambda m: m.group(1), text)
    text = INLINE_CODE_RE.sub("", text)
    text = re.sub(r"https?://\S+", "", text)
    return text


def filter_translatable_lines(lines: list[str], kind: str) -> list[str]:
    """Remove code/math payload and quote prefixes before translation."""
    filtered: list[str] = []
    in_fence = False
    in_math = False
    for index, raw in enumerate(lines):
        line = strip_quote(raw)
        core = structural_core(raw)
        if FENCE_RE.match(core):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # A complete inline-math line can contain a ``$$`` sequence where one
        # inline span closes immediately before the next opens (``$...$$...$``).
        # That sequence is not a display-math delimiter.  Standalone formula
        # lines are structural content and must remain untranslated anyway.
        stripped_line = line.strip()
        if stripped_line.startswith("$") and stripped_line.endswith("$"):
            continue
        # Display-math delimiters are sometimes attached to prose in the
        # source notes (for example ``explanation: $$`` or ``$$ formula $$
        # conclusion``).  Preserve only the prose portions while tracking the
        # delimiter parity; treating only a line equal to ``$$`` as a boundary
        # can otherwise swallow several later headings into one translation
        # unit.
        parts = line.split("$$")
        prose_parts: list[str] = []
        for part_index, part in enumerate(parts):
            if not in_math:
                prose_parts.append(part)
            if part_index < len(parts) - 1:
                in_math = not in_math
        line = " ".join(part.strip() for part in prose_parts if part.strip())
        if not line.strip():
            continue
        if re.fullmatch(r"\s*\^[A-Za-z0-9_-]+\s*", line):
            continue
        if kind == "callout" and index == 0 and re.match(r"^\[![^\]]+\]", line):
            # The callout declaration is structural UI, not body prose.  Its
            # visible title stays in Chinese; repeating a translated title as
            # a bold first body line produces a misleading duplicate heading
            # in Obsidian reading view.
            continue
        filtered.append(line)
    while filtered and not filtered[0].strip():
        filtered.pop(0)
    while filtered and not filtered[-1].strip():
        filtered.pop()
    return filtered


def classify(lines: list[str]) -> str:
    first = next((line for line in lines if line.strip()), "")
    plain = strip_quote(first).lstrip()
    if re.match(r"^\[![^\]]+\]", plain):
        return "callout"
    if plain.startswith("|"):
        return "table"
    if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", plain):
        return "list"
    if QUOTE_RE.match(first):
        return "quote"
    return "paragraph"


def common_quote_prefix(lines: list[str]) -> str:
    prefixes = []
    for line in lines:
        if not line.strip():
            continue
        match = QUOTE_RE.match(line)
        if not match:
            return ""
        prefixes.append(match.group(1))
    return min(prefixes, key=len) if prefixes else ""


def collect_units(lines: list[str]) -> list[dict]:
    units: list[dict] = []
    index = 0
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    if in_frontmatter:
        index = 1
        while index < len(lines):
            if lines[index].strip() == "---":
                index += 1
                break
            index += 1

    unit_id = 1
    while index < len(lines):
        line = lines[index]
        heading = HEADING_RE.match(line)
        if heading:
            if CJK_RE.search(visible_text(heading.group(2))):
                units.append(
                    {
                        "id": unit_id,
                        "kind": "heading",
                        "start": index,
                        "end": index,
                        "text": heading.group(2),
                        "quote_prefix": "",
                    }
                )
                unit_id += 1
            index += 1
            continue

        core = structural_core(line)
        if FENCE_RE.match(core):
            if core.count("```") >= 2 or core.count("~~~") >= 2:
                index += 1
                continue
            fence_token = core[:3]
            index += 1
            while index < len(lines):
                if structural_core(lines[index]).startswith(fence_token):
                    index += 1
                    break
                index += 1
            continue
        if core == "$$":
            index += 1
            while index < len(lines):
                if structural_core(lines[index]) == "$$":
                    index += 1
                    break
                index += 1
            continue
        if not line.strip():
            index += 1
            continue

        start = index
        block: list[str] = []
        nested_fence = False
        nested_math = False
        while index < len(lines):
            current = lines[index]
            if index > start and HEADING_RE.match(current) and not nested_fence and not nested_math:
                break
            current_core = structural_core(current)
            if FENCE_RE.match(current_core):
                if current_core.count("```") < 2 and current_core.count("~~~") < 2:
                    nested_fence = not nested_fence
            elif not nested_fence:
                display_count = current_core.count("$$")
                # Treat an odd delimiter as a cross-line display boundary only
                # when it occurs at a line boundary.  A lone ``$$`` embedded
                # in ``$...$$...$`` is just two adjacent inline-math markers.
                if display_count % 2 == 1 and (
                    current_core.startswith("$$") or current_core.endswith("$$")
                ):
                    nested_math = not nested_math
            if not current.strip() and not nested_fence and not nested_math:
                break
            block.append(current)
            index += 1

        kind = classify(block)
        translatable = filter_translatable_lines(block, kind)
        source = "\n".join(translatable)
        if source.strip() and CJK_RE.search(visible_text(source)):
            units.append(
                {
                    "id": unit_id,
                    "kind": kind,
                    "start": start,
                    "end": index - 1,
                    "text": source,
                    "quote_prefix": common_quote_prefix(block)
                    if kind in {"callout", "quote"}
                    else "",
                }
            )
            unit_id += 1
    return units


SYSTEM_PROMPT = """You are a meticulous Chinese-to-English translator for university-level academic notes. Return JSON only.

For every input unit, provide one faithful, idiomatic English Markdown translation with the same id and kind.
- Translate every visible Chinese statement; do not summarize, omit, add commentary, or leave Chinese visible.
- Preserve the meaning and order. Keep existing English technical terms when they are already natural, and use standard terminology for the note's actual academic field.
- Preserve inline-code spans, formulas, URLs, Markdown emphasis, list/checklist/table structure, and link destinations.
- For a wikilink whose destination contains Chinese, keep that destination but provide an idiomatic English display label: [[中文目标|English label]].
- Do not output bilingual markers, Markdown headings, callout declarations, code fences, YAML, or block IDs.
- A heading translation is plain title text; the caller will add italics.
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "kind": {"type": "string"},
                    "english": {"type": "string"},
                },
                "required": ["id", "kind", "english"],
            },
        }
    },
    "required": ["translations"],
}


def list_item_count(text: str) -> int:
    return sum(
        bool(re.match(r"^\s*(?:[-*+]\s+(?:\[[ xX]\]\s+)?|\d+[.)]\s+)", line))
        for line in text.splitlines()
    )


def preserved_tokens(text: str) -> dict[str, list[str]]:
    wiki_targets = []
    for match in WIKILINK_RE.finditer(text):
        if match.group(0).startswith("!"):
            continue
        spec = match.group(1)
        wiki_targets.append(spec.split("|", 1)[0])
    markdown_targets = re.findall(r"!?\[[^\]]*\]\(([^)]*)\)", text)
    return {
        "code": sorted(INLINE_CODE_RE.findall(text)),
        "math": sorted(re.findall(r"(?<!\$)\$[^$\n]+\$(?!\$)", text)),
        "wiki": sorted(wiki_targets),
        "markdown": sorted(markdown_targets),
    }


def google_request(text: str) -> str:
    if not text.strip():
        return text
    payload = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": "zh-CN",
            "tl": "en",
            "dt": "t",
            "q": text,
        }
    ).encode()
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                "https://translate.googleapis.com/translate_a/single",
                data=payload,
                timeout=60,
            ) as response:
                result = json.load(response)
            return "".join(part[0] or "" for part in result[0])
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def split_translation_chunks(text: str, limit: int = 3500) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces = re.split(r"(\n+|(?<=[。！？；])\s*)", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= limit:
            current += piece
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(piece) > limit:
            chunks.append(piece[:limit])
            piece = piece[limit:]
        current = piece
    if current:
        chunks.append(current)
    return chunks


def prepare_machine_unit(
    unit: dict, translate_label, protect_numbers: bool = True
) -> tuple[str, dict[str, str]]:
    text = unit["text"]
    replacements: dict[str, str] = {}

    def token(value: str) -> str:
        number = len(replacements)
        first = PLACEHOLDER_WORDS[number % len(PLACEHOLDER_WORDS)]
        cycle = number // len(PLACEHOLDER_WORDS)
        suffix = "" if cycle == 0 else PLACEHOLDER_WORDS[cycle - 1]
        key = f"XQJ{first}{suffix}JQX"
        replacements[key] = value
        return key

    def hundred_million(match: re.Match[str]) -> str:
        amount = Decimal(match.group(1).replace(",", ""))
        if amount >= 10000:
            value = amount / Decimal(10000)
            rendered = f"{value.normalize():f} trillion yuan"
        elif amount >= 10:
            value = amount / Decimal(10)
            rendered = f"{value.normalize():f} billion yuan"
        else:
            value = amount * Decimal(100)
            rendered = f"{value.normalize():f} million yuan"
        return token(rendered)

    def wiki(match: re.Match[str]) -> str:
        if match.group(0).startswith("!"):
            return ""
        spec = match.group(1)
        target, display = (spec.split("|", 1) + [None])[:2]
        label = display or Path(target.split("#", 1)[0]).stem
        if CJK_RE.search(label):
            label = translate_label(label).strip()
        return token(f"[[{target}|{label}]]")

    def markdown_link(match: re.Match[str]) -> str:
        whole = match.group(0)
        if whole.startswith("!"):
            return ""
        label, destination = re.match(r"\[([^\]]*)\]\(([^)]*)\)", whole).groups()
        if CJK_RE.search(label):
            label = translate_label(label).strip()
        return token(f"[{label}]({destination})")

    text = WIKILINK_RE.sub(wiki, text)
    text = re.sub(r"!?\[[^\]]*\]\([^)]*\)", markdown_link, text)
    text = INLINE_CODE_RE.sub(lambda m: token(m.group(0)), text)
    text = re.sub(
        r"(?<!\$)\$[^$\n]+\$(?!\$)",
        lambda m: token(m.group(0)),
        text,
    )
    for source, target in sorted(TERM_GLOSSARY, key=lambda item: -len(item[0])):
        if source in text:
            text = text.replace(source, token(target))
    if protect_numbers:
        text = re.sub(
            r"(\d+(?:[,.]\d+)?)\s*亿元(?:人民币)?", hundred_million, text
        )
        text = re.sub(
            r"\d+(?:[,.]\d+)*", lambda match: token(match.group(0)), text
        )
    text = re.sub(r"https?://\S+", lambda m: token(m.group(0)), text)
    return text, replacements


def restore_machine_unit(english: str, replacements: dict[str, str]) -> str:
    for key, value in reversed(list(replacements.items())):
        english, count = re.subn(
            re.escape(key), lambda _: value, english, flags=re.IGNORECASE
        )
        if count == 0:
            core = key.removeprefix("XQJ").removesuffix("JQX")
            for damaged in (
                rf"XQJ+{re.escape(core)}J?Q?X",
                rf"XQJ{re.escape(core)}\b",
                rf"\b{re.escape(core)}JQX",
                rf"\b{re.escape(core)}\b",
            ):
                english, restored = re.subn(
                    damaged,
                    lambda _: value,
                    english,
                    count=1,
                    flags=re.IGNORECASE,
                )
                if restored:
                    break

    # Qwen occasionally removes the boundary between two adjacent protected
    # tokens, for example XQJALPHAJQX + XQJBRAVOJQX becomes
    # XQJALPHABRAVOJQX.  Recover only concatenations of consecutive tokens from
    # this unit; that constraint avoids guessing at arbitrary leftover text.
    replacement_items = list(replacements.items())
    replacement_cores = [
        key.removeprefix("XQJ").removesuffix("JQX").upper()
        for key, _ in replacement_items
    ]

    def restore_compound(match: re.Match[str]) -> str:
        core = match.group(1).upper()
        for start in range(len(replacement_items)):
            combined = ""
            for stop in range(start, len(replacement_items)):
                combined += replacement_cores[stop]
                if len(combined) > len(core):
                    break
                if combined == core and stop > start:
                    return " ".join(
                        value for _, value in replacement_items[start : stop + 1]
                    )
        return match.group(0)

    english = re.sub(
        r"XQJ([A-Z]+)JQX",
        restore_compound,
        english,
        flags=re.IGNORECASE,
    )
    if re.search(r"XQJ[A-Z]+JQX", english, flags=re.IGNORECASE):
        raise RuntimeError(
            "machine translator left an unrestored placeholder: " + repr(english[:500])
        )
    return english.strip()


def machine_translate_unit(unit: dict, translate) -> str:
    text, replacements = prepare_machine_unit(unit, translate)
    english = "".join(translate(chunk) for chunk in split_translation_chunks(text))
    return restore_machine_unit(english, replacements)


def request_google_translations(units: list[dict]) -> dict[int, str]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        english = list(
            pool.map(lambda unit: machine_translate_unit(unit, google_request), units)
        )
    translated = {unit["id"]: text for unit, text in zip(units, english)}
    failures = {}
    for unit in units:
        issues = [
            issue
            for issue in translation_issues(unit, translated[unit["id"]])
            if not issue.startswith("list-item count differs")
            and issue != "heading number was not preserved"
        ]
        if issues:
            failures[unit["id"]] = issues
    if failures:
        # Keep the complete draft available for the required human QA pass.
        # make_content() will still reject visible CJK, so blocking units must
        # be repaired before anything can be inserted.
        print(
            "GOOGLE_WARN " + repr(failures),
            file=sys.stderr,
            flush=True,
        )
    return translated


def request_argos_translations(units: list[dict]) -> dict[int, str]:
    try:
        import argostranslate.translate
    except ImportError as exc:
        raise RuntimeError(
            "Argos Translate is unavailable in this Python environment; "
            "run the script with /opt/anaconda3/bin/python"
        ) from exc

    source = next(
        language
        for language in argostranslate.translate.get_installed_languages()
        if language.code == "zh"
    )
    target = next(
        language
        for language in argostranslate.translate.get_installed_languages()
        if language.code == "en"
    )
    translator = source.get_translation(target)
    translated = {
        unit["id"]: machine_translate_unit(unit, translator.translate) for unit in units
    }
    failures = {}
    for unit in units:
        issues = [
            issue
            for issue in translation_issues(unit, translated[unit["id"]])
            if not issue.startswith("list-item count differs")
            and issue != "heading number was not preserved"
        ]
        if issues:
            failures[unit["id"]] = issues
    if failures:
        raise RuntimeError(f"Argos translation validation failed: {failures}")
    return translated


_NLLB_STATE = None


def nllb_translate_many(texts: list[str], batch_size: int = 12) -> list[str]:
    global _NLLB_STATE
    if not texts:
        return []
    if _NLLB_STATE is None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLLB dependencies are unavailable; run with /opt/anaconda3/bin/python"
            ) from exc
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="zho_Hans")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        model.eval()
        _NLLB_STATE = (torch, tokenizer, model, device)

    torch, tokenizer, model, device = _NLLB_STATE
    output: list[str | None] = [None] * len(texts)
    ordered = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    for start in range(0, len(ordered), batch_size):
        indices = ordered[start : start + batch_size]
        batch = [texts[index] for index in indices]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=384,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for index, translation in zip(indices, decoded):
            output[index] = translation.strip()
    return [text or "" for text in output]


def request_nllb_translations(units: list[dict]) -> dict[int, str]:
    labels: list[str] = []
    for unit in units:
        for match in WIKILINK_RE.finditer(unit["text"]):
            if match.group(0).startswith("!"):
                continue
            spec = match.group(1)
            target, display = (spec.split("|", 1) + [None])[:2]
            label = display or Path(target.split("#", 1)[0]).stem
            if CJK_RE.search(label):
                labels.append(label)
        for match in re.finditer(r"(?<!!)\[([^\]]*)\]\(([^)]*)\)", unit["text"]):
            label = match.group(1)
            if CJK_RE.search(label):
                labels.append(label)
    unique_labels = list(dict.fromkeys(labels))
    label_map = dict(zip(unique_labels, nllb_translate_many(unique_labels)))

    prepared: dict[int, tuple[str, dict[str, str]]] = {}
    tasks: list[str] = []
    task_keys: list[tuple[int, int, int]] = []
    line_chunks: dict[tuple[int, int], int] = {}
    for unit in units:
        text, replacements = prepare_machine_unit(
            unit,
            lambda label: label_map.get(label, label),
        )
        prepared[unit["id"]] = (text, replacements)
        for line_index, line in enumerate(text.splitlines()):
            if not line.strip():
                line_chunks[(unit["id"], line_index)] = 0
                continue
            chunks = split_translation_chunks(line, limit=450)
            line_chunks[(unit["id"], line_index)] = len(chunks)
            for chunk_index, chunk in enumerate(chunks):
                tasks.append(chunk)
                task_keys.append((unit["id"], line_index, chunk_index))

    translated_tasks = nllb_translate_many(tasks)
    translated_by_key = dict(zip(task_keys, translated_tasks))
    translated: dict[int, str] = {}
    for unit in units:
        text, replacements = prepared[unit["id"]]
        lines: list[str] = []
        for line_index, source_line in enumerate(text.splitlines()):
            chunk_count = line_chunks[(unit["id"], line_index)]
            if chunk_count == 0:
                lines.append("")
                continue
            pieces = [
                translated_by_key[(unit["id"], line_index, chunk_index)]
                for chunk_index in range(chunk_count)
            ]
            lines.append(" ".join(piece for piece in pieces if piece))
        translated[unit["id"]] = restore_machine_unit(
            "\n".join(lines), replacements
        )

    failures = {}
    for unit in units:
        issues = translation_issues(unit, translated[unit["id"]])
        if issues:
            failures[unit["id"]] = issues
    if failures:
        raise RuntimeError(f"NLLB translation validation failed: {failures}")
    return translated


def request_sogou_batched_translations(units: list[dict]) -> dict[int, str]:
    try:
        import translators as translators_api
    except ImportError as exc:
        raise RuntimeError(
            "The Sogou draft translator is unavailable; "
            "run with /opt/anaconda3/bin/python"
        ) from exc

    label_cache: dict[str, str] = {}

    def request(text: str) -> str:
        for attempt in range(5):
            try:
                return translators_api.translate_text(
                    text,
                    translator="sogou",
                    from_language="zh",
                    to_language="en",
                ).strip()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(20, 2**attempt))
        raise AssertionError("unreachable")

    def translate_label(label: str) -> str:
        if label not in label_cache:
            label_cache[label] = request(label)
        return label_cache[label]

    prepared: dict[int, tuple[str, dict[str, str]]] = {
        unit["id"]: prepare_machine_unit(unit, translate_label) for unit in units
    }
    segments: list[tuple[int, int, int, str]] = []
    line_chunks: dict[tuple[int, int], int] = {}
    for unit in units:
        text, _ = prepared[unit["id"]]
        for line_index, line in enumerate(text.splitlines()):
            if not line.strip():
                line_chunks[(unit["id"], line_index)] = 0
                continue
            chunks = split_translation_chunks(line, limit=1800)
            line_chunks[(unit["id"], line_index)] = len(chunks)
            for chunk_index, chunk in enumerate(chunks):
                segments.append((unit["id"], line_index, chunk_index, chunk))

    translated_segments: dict[tuple[int, int, int], str] = {}
    cursor = 0
    segment_number = 0
    while cursor < len(segments):
        batch: list[tuple[int, int, int, str, str]] = []
        batch_length = 0
        while cursor < len(segments) and len(batch) < 24:
            unit_id, line_index, chunk_index, source = segments[cursor]
            marker = f"__SEG{segment_number:06d}__"
            rendered = marker + source
            if batch and batch_length + len(rendered) + 1 > 4200:
                break
            batch.append((unit_id, line_index, chunk_index, marker, source))
            batch_length += len(rendered) + 1
            cursor += 1
            segment_number += 1

        payload = "\n".join(marker + source for _, _, _, marker, source in batch)
        response = request(payload)
        marker_pattern = re.compile(r"__SEG(\d{6})__")
        matches = list(marker_pattern.finditer(response))
        if len(matches) != len(batch):
            raise RuntimeError(
                "Sogou translation changed segment markers: "
                f"expected={len(batch)} actual={len(matches)}"
            )
        for index, item in enumerate(batch):
            start = matches[index].end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
            translated = response[start:end].strip()
            translated = re.sub(r"^(\s*[-+])(?=\S)", r"\1 ", translated)
            translated = re.sub(r"^(\s*\*)(?=[^*\s])", r"\1 ", translated)
            translated_segments[item[:3]] = translated

    translated: dict[int, str] = {}
    for unit in units:
        text, replacements = prepared[unit["id"]]
        lines: list[str] = []
        for line_index, _ in enumerate(text.splitlines()):
            chunk_count = line_chunks[(unit["id"], line_index)]
            if chunk_count == 0:
                lines.append("")
                continue
            pieces = [
                translated_segments[(unit["id"], line_index, chunk_index)]
                for chunk_index in range(chunk_count)
            ]
            lines.append(" ".join(piece for piece in pieces if piece))
        translated[unit["id"]] = restore_machine_unit(
            "\n".join(lines), replacements
        )

    failures = {}
    for unit in units:
        issues = translation_issues(unit, translated[unit["id"]])
        if issues:
            failures[unit["id"]] = issues
    if failures:
        raise RuntimeError(f"Sogou translation validation failed: {failures}")
    return translated


def request_sogou_translations(units: list[dict]) -> dict[int, str]:
    try:
        import translators as translators_api
    except ImportError as exc:
        raise RuntimeError(
            "The Sogou draft translator is unavailable; "
            "run with /opt/anaconda3/bin/python"
        ) from exc

    cache: dict[str, str] = {}

    def request(text: str) -> str:
        if text in cache:
            return cache[text]
        for attempt in range(5):
            try:
                translated = translators_api.translate_text(
                    text,
                    translator="sogou",
                    from_language="zh",
                    to_language="en",
                ).strip()
                translated = re.sub(
                    r"^(\s*[-+])(?=\S)", r"\1 ", translated, flags=re.MULTILINE
                )
                translated = re.sub(
                    r"^(\s*\*)(?=[^*\s])", r"\1 ", translated, flags=re.MULTILINE
                )
                cache[text] = translated
                return translated
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(20, 2**attempt))
        raise AssertionError("unreachable")

    translated = {
        unit["id"]: machine_translate_unit(unit, request) for unit in units
    }
    failures = {}
    for unit in units:
        issues = translation_issues(unit, translated[unit["id"]])
        if issues:
            failures[unit["id"]] = issues
    if failures:
        raise RuntimeError(f"Sogou translation validation failed: {failures}")
    return translated


def request_caiyun_translations(units: list[dict]) -> dict[int, str]:
    try:
        import translators as translators_api
    except ImportError as exc:
        raise RuntimeError(
            "The Caiyun draft translator is unavailable; "
            "run with /opt/anaconda3/bin/python"
        ) from exc

    cache: dict[str, str] = {}

    def request(text: str) -> str:
        if text in cache:
            return cache[text]
        for attempt in range(5):
            try:
                translated = translators_api.translate_text(
                    text,
                    translator="caiyun",
                    from_language="zh",
                    to_language="en",
                ).strip()
                translated = re.sub(r"\*\s+\*", "**", translated)
                translated = re.sub(
                    r"^(\s*[-+])(?=\S)", r"\1 ", translated, flags=re.MULTILINE
                )
                translated = re.sub(
                    r"^(\s*\*)(?=[^*\s])", r"\1 ", translated, flags=re.MULTILINE
                )
                cache[text] = translated
                return translated
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(20, 2**attempt))
        raise AssertionError("unreachable")

    translated = {
        unit["id"]: machine_translate_unit(unit, request) for unit in units
    }
    failures = {}
    for unit in units:
        issues = translation_issues(unit, translated[unit["id"]])
        if issues:
            failures[unit["id"]] = issues
    if failures:
        raise RuntimeError(f"Caiyun translation validation failed: {failures}")
    return translated


def request_qq_translations(units: list[dict]) -> dict[int, str]:
    try:
        import translators as translators_api
    except ImportError as exc:
        raise RuntimeError(
            "The QQ draft translator is unavailable; "
            "run with /opt/anaconda3/bin/python"
        ) from exc

    marker_words = [
        "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT",
        "GOLF", "HOTEL", "INDIA", "JULIET", "KILO", "LIMA", "MIKE",
        "NOVEMBER", "OSCAR", "PAPA", "QUEBEC", "ROMEO", "SIERRA",
        "TANGO", "UNIFORM", "VICTOR", "WHISKEY", "XRAY",
    ]
    markers = [f"VZQ{word}QZV" for word in marker_words]
    marker_pattern = re.compile(
        "(" + "|".join(map(re.escape, markers)) + ")", re.IGNORECASE
    )

    def request(text: str) -> str:
        for attempt in range(5):
            try:
                translated = translators_api.translate_text(
                    text,
                    translator="qqTranSmart",
                    from_language="zh",
                    to_language="en",
                ).strip()
                translated = re.sub(r"\*\s+\*", "**", translated)
                translated = re.sub(
                    r"\*\*\s*([^*\n]+?)\s*\*\*",
                    lambda match: f"**{match.group(1).strip()}**",
                    translated,
                )
                translated = re.sub(
                    r"^(\s*[-+])(?=\S)", r"\1 ", translated, flags=re.MULTILINE
                )
                translated = re.sub(
                    r"^(\s*\*)(?=[^*\s])", r"\1 ", translated, flags=re.MULTILINE
                )
                return translated
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(30, 2**attempt))
        raise AssertionError("unreachable")

    def translate_marked(texts: list[str]) -> list[str]:
        def run_batch(batch: list[str]) -> list[str]:
            payload = "\n".join(
                marker + text for marker, text in zip(markers, batch)
            )
            response = request(payload)
            matches = list(marker_pattern.finditer(response))
            if len(matches) != len(batch):
                if len(batch) == 1:
                    return [request(batch[0])]
                midpoint = len(batch) // 2
                return run_batch(batch[:midpoint]) + run_batch(batch[midpoint:])
            result: list[str] = []
            for index in range(len(batch)):
                start = matches[index].end()
                end = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(response)
                )
                result.append(response[start:end].strip())
            return result

        output: list[str] = []
        cursor = 0
        while cursor < len(texts):
            batch: list[str] = []
            batch_length = 0
            while cursor + len(batch) < len(texts) and len(batch) < len(markers):
                candidate = texts[cursor + len(batch)]
                rendered_length = len(markers[len(batch)]) + len(candidate) + 1
                if batch and batch_length + rendered_length > 4300:
                    break
                batch.append(candidate)
                batch_length += rendered_length
            output.extend(run_batch(batch))
            cursor += len(batch)
        return output

    labels: list[str] = []
    for unit in units:
        for match in WIKILINK_RE.finditer(unit["text"]):
            if match.group(0).startswith("!"):
                continue
            spec = match.group(1)
            target, display = (spec.split("|", 1) + [None])[:2]
            label = display or Path(target.split("#", 1)[0]).stem
            if CJK_RE.search(label):
                labels.append(label)
        for match in re.finditer(r"(?<!!)\[([^\]]*)\]\(([^)]*)\)", unit["text"]):
            label = match.group(1)
            if CJK_RE.search(label):
                labels.append(label)
    unique_labels = list(dict.fromkeys(labels))
    label_map = dict(zip(unique_labels, translate_marked(unique_labels)))

    prepared: dict[int, tuple[list[str], dict[str, str]]] = {}
    task_keys: list[tuple[int, int]] = []
    tasks: list[str] = []
    for unit in units:
        text, replacements = prepare_machine_unit(
            unit,
            lambda label: label_map.get(label, label),
            protect_numbers=False,
        )
        chunks = split_translation_chunks(text, limit=1800)
        prepared[unit["id"]] = (chunks, replacements)
        for chunk_index, chunk in enumerate(chunks):
            task_keys.append((unit["id"], chunk_index))
            tasks.append(chunk)

    translated_tasks = dict(zip(task_keys, translate_marked(tasks)))
    translated: dict[int, str] = {}
    for unit in units:
        chunks, replacements = prepared[unit["id"]]
        english = "".join(
            translated_tasks[(unit["id"], chunk_index)]
            for chunk_index in range(len(chunks))
        )
        translated[unit["id"]] = restore_machine_unit(english, replacements)

    failures = {}
    for unit in units:
        issues = translation_issues(unit, translated[unit["id"]])
        if issues:
            failures[unit["id"]] = issues
    if failures:
        raise RuntimeError(f"QQ translation validation failed: {failures}")
    return translated


def request_qq_sequential_translations(units: list[dict]) -> dict[int, str]:
    try:
        import translators as translators_api
    except ImportError as exc:
        raise RuntimeError(
            "The QQ draft translator is unavailable; "
            "run with /opt/anaconda3/bin/python"
        ) from exc

    cache: dict[str, str] = {}

    def request(text: str) -> str:
        if text in cache:
            return cache[text]
        for attempt in range(5):
            try:
                translated = translators_api.translate_text(
                    text,
                    translator="qqTranSmart",
                    from_language="zh",
                    to_language="en",
                ).strip()
                translated = re.sub(r"\*\s+\*", "**", translated)
                translated = re.sub(
                    r"\*\*\s*([^*\n]+?)\s*\*\*",
                    lambda match: f"**{match.group(1).strip()}**",
                    translated,
                )
                translated = re.sub(
                    r"^(\s*[-+])(?=\S)", r"\1 ", translated, flags=re.MULTILINE
                )
                translated = re.sub(
                    r"^(\s*\*)(?=[^*\s])", r"\1 ", translated, flags=re.MULTILINE
                )
                cache[text] = translated
                return translated
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(30, 2**attempt))
        raise AssertionError("unreachable")

    translated: dict[int, str] = {}
    translation_errors: dict[int, str] = {}
    for unit in units:
        try:
            translated[unit["id"]] = machine_translate_unit(unit, request)
        except RuntimeError as exc:
            translated[unit["id"]] = ""
            translation_errors[unit["id"]] = str(exc)
    failures = {}
    for unit in units:
        issues = translation_issues(unit, translated[unit["id"]])
        if issues:
            failures[unit["id"]] = issues
    if failures or translation_errors:
        print(
            "QQ_WARN "
            + repr({"validation": failures, "errors": translation_errors}),
            file=sys.stderr,
            flush=True,
        )
    return translated


_QWEN_STATE = None


def qwen_translate_many(texts: list[str]) -> list[str]:
    global _QWEN_STATE
    if not texts:
        return []
    if _QWEN_STATE is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen dependencies are unavailable; run with /opt/anaconda3/bin/python"
            ) from exc
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        model.eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        _QWEN_STATE = (torch, tokenizer, model, device)

    torch, tokenizer, model, device = _QWEN_STATE
    system = (
        "Translate Chinese university-level academic notes into faithful, "
        "idiomatic English. Preserve every existing English term, number, "
        "formula, Markdown marker, line order, and list item. Tokens beginning "
        "with XQJ and ending with JQX are protected and must be copied exactly. "
        "Do not explain, summarise, or add content. Output only the translation."
    )
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in texts
    ]
    lengths = [len(tokenizer(prompt).input_ids) for prompt in prompts]
    ordered = sorted(range(len(texts)), key=lambda index: lengths[index])
    output: list[str | None] = [None] * len(texts)
    cursor = 0
    while cursor < len(ordered):
        next_length = lengths[ordered[cursor]]
        if next_length <= 180:
            batch_size = 32
        elif next_length <= 360:
            batch_size = 20
        elif next_length <= 700:
            batch_size = 10
        elif next_length <= 1300:
            batch_size = 4
        else:
            batch_size = 2
        indices = ordered[cursor : cursor + batch_size]
        batch_prompts = [prompts[index] for index in indices]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)
        maximum = max(lengths[index] for index in indices)
        max_new_tokens = min(400, max(96, int(maximum * 0.9)))
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        decoded = tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        for index, translation in zip(indices, decoded):
            output[index] = translation.strip()
        cursor += len(indices)
    return [text or "" for text in output]


def request_qwen_translations(units: list[dict]) -> dict[int, str]:
    labels: list[str] = []
    for unit in units:
        for match in WIKILINK_RE.finditer(unit["text"]):
            if match.group(0).startswith("!"):
                continue
            spec = match.group(1)
            target, display = (spec.split("|", 1) + [None])[:2]
            label = display or Path(target.split("#", 1)[0]).stem
            if CJK_RE.search(label):
                labels.append(label)
        for match in re.finditer(r"(?<!!)\[([^\]]*)\]\(([^)]*)\)", unit["text"]):
            label = match.group(1)
            if CJK_RE.search(label):
                labels.append(label)
    unique_labels = list(dict.fromkeys(labels))
    translated_labels = [
        normalize_english(label, "heading")
        for label in qwen_translate_many(unique_labels)
    ]
    label_map = dict(zip(unique_labels, translated_labels))

    prepared: dict[int, tuple[list[str], dict[str, str]]] = {}
    task_keys: list[tuple[int, int]] = []
    tasks: list[str] = []
    for unit in units:
        text, replacements = prepare_machine_unit(
            unit,
            lambda label: label_map.get(label, label),
            protect_numbers=False,
        )
        chunks = split_translation_chunks(text, limit=300)
        prepared[unit["id"]] = (chunks, replacements)
        for chunk_index, chunk in enumerate(chunks):
            task_keys.append((unit["id"], chunk_index))
            tasks.append(chunk)

    translated_tasks = dict(zip(task_keys, qwen_translate_many(tasks)))
    translated: dict[int, str] = {}
    for unit in units:
        chunks, replacements = prepared[unit["id"]]
        english = "".join(
            translated_tasks[(unit["id"], chunk_index)]
            for chunk_index in range(len(chunks))
        )
        translated[unit["id"]] = restore_machine_unit(english, replacements)

    retry_ids = [
        unit["id"]
        for unit in units
        if CJK_RE.search(visible_text(translated[unit["id"]]))
        or CHINESE_PUNCT_RE.search(visible_text(translated[unit["id"]]))
    ]
    retry_payloads: list[str] = []
    retry_replacements: dict[int, dict[str, str]] = {}
    for unit_id in retry_ids:
        pseudo_unit = {"text": translated[unit_id]}
        payload, replacements = prepare_machine_unit(
            pseudo_unit, lambda label: label, protect_numbers=False
        )
        retry_payloads.append(payload)
        retry_replacements[unit_id] = replacements
    for unit_id, retry in zip(retry_ids, qwen_translate_many(retry_payloads)):
        try:
            translated[unit_id] = restore_machine_unit(
                retry, retry_replacements[unit_id]
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Qwen retry unit {unit_id}: {exc}") from exc

    # A rare short mixed Chinese/English sentence can be echoed by the local
    # model even after the focused retry.  Use the already validated sequential
    # QQ path only for those residual units instead of discarding the rest of a
    # successfully translated file.
    fallback_units = [
        unit
        for unit in units
        if CJK_RE.search(visible_text(translated[unit["id"]]))
        or CHINESE_PUNCT_RE.search(visible_text(translated[unit["id"]]))
        or re.search(r"XQJ|JQX", visible_text(translated[unit["id"]]), re.IGNORECASE)
    ]
    if fallback_units:
        try:
            translated.update(request_qq_sequential_translations(fallback_units))
        except RuntimeError as exc:
            # Preserve the otherwise successful Qwen batch for manual repair.
            # QQ occasionally damages a protected token in a long formula-heavy
            # unit; failing the whole file would discard every good draft.
            print(
                "QWEN_FALLBACK_WARN "
                + repr([unit["id"] for unit in fallback_units])
                + ": "
                + str(exc),
                file=sys.stderr,
            )

    failures = {}
    warnings = {}
    for unit in units:
        source_number_text = WIKILINK_RE.sub("", unit["text"])
        english_number_text = WIKILINK_RE.sub("", translated[unit["id"]])
        source_numbers = [
            value.replace(",", "")
            for value in re.findall(r"\d+(?:[,.]\d+)*", source_number_text)
        ]
        if "![[" in unit["text"] and len(source_numbers) > 20:
            translated[unit["id"]] = translated[unit["id"]].replace("$", "USD ")
            english_number_text = WIKILINK_RE.sub("", translated[unit["id"]])
        issues = translation_issues(unit, translated[unit["id"]])
        english_numbers = [
            value.replace(",", "")
            for value in re.findall(r"\d+(?:[,.]\d+)*", english_number_text)
        ]
        if not ("![[" in unit["text"] and len(source_numbers) > 20) and (
            source_numbers != english_numbers
        ):
            issues.append("numeric values or order changed")
        blocking = [
            issue
            for issue in issues
            if issue
            in {
                "empty translation",
                "visible Chinese characters remain",
                "Chinese punctuation remains",
                "protected placeholder residue remains",
            }
        ]
        if blocking:
            failures[unit["id"]] = blocking
        if issues:
            warnings[unit["id"]] = issues
    if failures:
        # Return the complete draft so the caller can repair the small number
        # of blocking units without rerunning an expensive successful batch.
        # make_content() still rejects visible CJK, so an unreviewed draft
        # cannot be inserted accidentally.
        print(
            "QWEN_BLOCKING_WARN " + repr(failures),
            file=sys.stderr,
            flush=True,
        )
    if warnings:
        print(
            "QWEN_WARN " + repr(warnings),
            file=sys.stderr,
            flush=True,
        )
    return translated


def translation_issues(unit: dict, english: str) -> list[str]:
    english = normalize_english(english, unit["kind"])
    issues: list[str] = []
    visible = visible_text(english)
    if not english:
        issues.append("empty translation")
    if CJK_RE.search(visible):
        issues.append("visible Chinese characters remain")
    if CHINESE_PUNCT_RE.search(visible):
        issues.append("Chinese punctuation remains")
    if re.search(r"XQJ|JQX", visible, flags=re.IGNORECASE):
        issues.append("protected placeholder residue remains")
    source_items = 0 if unit["kind"] == "heading" else list_item_count(unit["text"])
    english_items = 0 if unit["kind"] == "heading" else list_item_count(english)
    if source_items != english_items:
        issues.append(
            f"list-item count differs: source={source_items}, translation={english_items}"
        )
    source_rows = sum(line.lstrip().startswith("|") for line in unit["text"].splitlines())
    english_rows = sum(line.lstrip().startswith("|") for line in english.splitlines())
    if source_rows != english_rows:
        issues.append(
            f"table-row count differs: source={source_rows}, translation={english_rows}"
        )
    if unit["kind"] == "heading":
        source_number = re.match(r"^(\d+)\.", unit["text"])
        english_number = re.match(r"^(\d+)\.", english)
        if bool(source_number) != bool(english_number) or (
            source_number
            and english_number
            and source_number.group(1) != english_number.group(1)
        ):
            issues.append("heading number was not preserved")
        if "\n" in english:
            issues.append("heading translation spans multiple lines")
    source_tokens = preserved_tokens(unit["text"])
    english_tokens = preserved_tokens(english)
    for token_kind in source_tokens:
        if token_kind == "code":
            changed = bool(
                Counter(source_tokens[token_kind]) - Counter(english_tokens[token_kind])
            )
        else:
            changed = source_tokens[token_kind] != english_tokens[token_kind]
        if changed:
            issues.append(f"{token_kind} tokens or destinations changed")
    return issues


def request_translation_batch(
    units: list[dict],
    model: str,
    previous: dict[int, str] | None = None,
    issue_map: dict[int, list[str]] | None = None,
) -> dict[int, str]:
    request_units = [
        {
            "id": unit["id"],
            "kind": unit["kind"],
            "text": unit["text"],
            "expected_list_items": 0
            if unit["kind"] == "heading"
            else list_item_count(unit["text"]),
            **(
                {
                    "previous_draft": previous.get(unit["id"], ""),
                    "validation_errors": (issue_map or {}).get(unit["id"], []),
                }
                if previous is not None
                else {}
            ),
        }
        for unit in units
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": (
                            "Translate these units. Match every bullet and sentence."
                            if previous is None
                            else "Revise every previous draft to fix all listed validation errors. "
                            "Return complete, idiomatic English translations, not patches."
                        ),
                        "units": request_units,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": 32768},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    parsed = json.loads(result["message"]["content"])
    translated = parsed.get("translations", [])
    by_id = {item["id"]: item["english"].strip() for item in translated}
    expected = {unit["id"] for unit in units}
    extra = sorted(set(by_id) - expected)
    if extra:
        raise RuntimeError(f"translation ids mismatch: extra={extra}")
    return by_id


def request_translations(units: list[dict], model: str) -> dict[int, str]:
    if model == "google-translate":
        return request_google_translations(units)
    if model == "argos-translate":
        return request_argos_translations(units)
    if model == "nllb-translate":
        return request_nllb_translations(units)
    if model == "sogou-translate":
        return request_sogou_translations(units)
    if model == "caiyun-translate":
        return request_caiyun_translations(units)
    if model == "qq-translate":
        return request_qq_translations(units)
    if model == "qq-sequential":
        return request_qq_sequential_translations(units)
    if model == "qwen-local":
        return request_qwen_translations(units)
    by_id: dict[int, str] = {}
    batch_size = 10
    for start in range(0, len(units), batch_size):
        batch = units[start : start + batch_size]
        batch_result = request_translation_batch(batch, model)
        by_id.update(batch_result)
        missing = [unit for unit in batch if unit["id"] not in batch_result]
        for _ in range(3):
            if not missing:
                break
            retry_result = request_translation_batch(missing, model)
            by_id.update(retry_result)
            missing = [unit for unit in missing if unit["id"] not in retry_result]
        if missing:
            raise RuntimeError(
                f"translation ids still missing after retries: "
                f"{[unit['id'] for unit in missing]}"
            )
        issue_map = {
            unit["id"]: translation_issues(unit, by_id[unit["id"]])
            for unit in batch
        }
        invalid = [unit for unit in batch if issue_map[unit["id"]]]
        for _ in range(3):
            if not invalid:
                break
            previous = {unit["id"]: by_id.get(unit["id"], "") for unit in invalid}
            retry_issues = {unit["id"]: issue_map[unit["id"]] for unit in invalid}
            retry_result = request_translation_batch(
                invalid, model, previous=previous, issue_map=retry_issues
            )
            by_id.update(retry_result)
            issue_map.update(
                {
                    unit["id"]: translation_issues(
                        unit, by_id.get(unit["id"], "")
                    )
                    for unit in invalid
                }
            )
            invalid = [unit for unit in invalid if issue_map[unit["id"]]]
        if invalid:
            print(
                "WARNING: draft translation still has validation issues; "
                "manual review is required: "
                + repr({unit["id"]: issue_map[unit["id"]] for unit in invalid}),
                file=sys.stderr,
            )
    return by_id


def normalize_english(text: str, kind: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.replace(START, "").replace(END, "").strip()
    if kind == "heading":
        text = re.sub(r"^#{1,6}\s+", "", text).strip()
        text = text.strip("*_ ")
        if "\n" in text:
            text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    return text


def visible_english_has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(visible_text(text)))


def build_insertion(unit: dict, english: str) -> list[str]:
    kind = unit["kind"]
    english = normalize_english(english, kind)
    if not english:
        raise RuntimeError(f"empty translation for unit {unit['id']}")
    if visible_english_has_cjk(english):
        raise RuntimeError(
            f"visible CJK remains in unit {unit['id']}: {english[:240]!r}"
        )
    if kind == "heading":
        body = [f"*{english}*"]
        prefix = ""
    else:
        body = english.splitlines()
        prefix = unit["quote_prefix"]
    if prefix:
        rendered = [prefix + START]
        rendered.extend(prefix.rstrip() if not line else prefix + line for line in body)
        rendered.append(prefix + END)
        return rendered
    return [START, *body, END]


def make_content(original: str, units: list[dict], translations: dict[int, str]) -> str:
    final_newline = original.endswith("\n")
    lines = original.splitlines()
    after: dict[int, list[str]] = {}
    for unit in units:
        if unit["end"] in after:
            raise RuntimeError(f"overlapping insertion at line {unit['end'] + 1}")
        after[unit["end"]] = build_insertion(unit, translations[unit["id"]])
    result: list[str] = []
    for index, line in enumerate(lines):
        result.append(line)
        if index in after:
            result.extend(after[index])
    output = "\n".join(result)
    if final_newline:
        output += "\n"
    return output


def patch_for(path: Path, content: str) -> str:
    original = path.read_text().splitlines()
    updated = content.splitlines()
    diff = list(
        difflib.unified_diff(
            original, updated, fromfile="", tofile="", n=3, lineterm=""
        )
    )[2:]
    body = "".join(("@@" if line.startswith("@@ ") else line) + "\n" for line in diff)
    return (
        "*** Begin Patch\n"
        f"*** Update File: {path.resolve()}\n"
        f"{body}"
        "*** End Patch\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", default="deepseek-r1:14b")
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()

    path = args.path
    baseline = Path("tmp/bilingual_baseline/classroom") / path
    if not baseline.exists():
        raise SystemExit(f"missing baseline: {baseline}")
    original = baseline.read_text()
    current = path.read_text()
    if START not in current and current.rstrip("\n") != original.rstrip("\n"):
        raise SystemExit(f"current file differs from baseline before bilingualization: {path}")
    if START not in current:
        original = current
    units = collect_units(original.splitlines())
    if args.inventory:
        print(json.dumps(units, ensure_ascii=False, indent=2))
        return
    translations = request_translations(units, args.model)
    content = make_content(original, units, translations)
    print(patch_for(path, content), end="")


if __name__ == "__main__":
    main()
