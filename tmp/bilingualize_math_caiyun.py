#!/usr/bin/env python3
"""Generate an apply_patch payload with Caiyun drafts for manual review."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
import time
from pathlib import Path

import translators as translators_api

sys.path.insert(0, str(Path(__file__).parent))
import bilingualize_classroom as shared


GLOSSARY = [
    ("协整", "cointegration"),
    ("共同随机趋势", "common stochastic trend"),
    ("误差修正模型", "error-correction model"),
    ("误差纠正模型", "error-correction model"),
    ("平稳随机过程", "stationary stochastic process"),
    ("平稳过程", "stationary process"),
    ("非平稳过程", "nonstationary process"),
    ("非平稳", "nonstationary"),
    ("单位根", "unit root"),
    ("随机过程", "stochastic process"),
    ("随机趋势", "stochastic trend"),
    ("协方差函数", "covariance function"),
    ("互协方差函数", "cross-covariance function"),
    ("互相关函数", "cross-correlation function"),
    ("相关函数", "correlation function"),
    ("均值函数", "mean function"),
    ("方差函数", "variance function"),
    ("正交增量过程", "orthogonal-increment process"),
    ("独立增量过程", "independent-increment process"),
    ("平稳独立增量", "stationary independent increments"),
    ("宽平稳", "weakly stationary"),
    ("严平稳", "strictly stationary"),
    ("多元正态分布", "multivariate normal distribution"),
    ("维纳过程", "Wiener process"),
    ("布朗运动", "Brownian motion"),
    ("条件异方差", "conditional heteroskedasticity"),
    ("波动率聚集", "volatility clustering"),
    ("极大似然估计", "maximum likelihood estimation"),
    ("脉冲响应函数", "impulse response function"),
    ("方差分解", "variance decomposition"),
]


def placeholder(index: int) -> str:
    digest = hashlib.sha256(f"math-bilingual-{index}".encode()).digest()
    body = "".join(chr(ord("A") + byte % 26) for byte in digest[:12])
    return "QZ" + body + "VQ"


class Translator:
    def __init__(self) -> None:
        self.cache: dict[str, str] = {}

    def request(self, text: str) -> str:
        if text in self.cache:
            return self.cache[text]
        for attempt in range(4):
            try:
                result = translators_api.translate_text(
                    text,
                    translator="caiyun",
                    from_language="zh",
                    to_language="en",
                ).strip()
                result = re.sub(r"\*\s+\*", "**", result)
                result = re.sub(r"^(\s*[-+])(?=\S)", r"\1 ", result, flags=re.M)
                result = re.sub(r"^(\s*\*)(?=[^*\s])", r"\1 ", result, flags=re.M)
                self.cache[text] = result
                return result
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def prepare(self, unit: dict) -> tuple[str, dict[str, str]]:
        text = unit["text"]
        replacements: dict[str, str] = {}

        def protect(value: str) -> str:
            key = placeholder(len(replacements))
            replacements[key] = value
            return key

        text = shared.INLINE_CODE_RE.sub(lambda match: protect(match.group(0)), text)
        text = re.sub(
            r"(?<!\$)\$[^$\n]+\$(?!\$)",
            lambda match: protect(match.group(0)),
            text,
        )

        def wiki(match: re.Match[str]) -> str:
            if match.group(0).startswith("!"):
                return ""
            spec = match.group(1)
            target, display = (spec.split("|", 1) + [None])[:2]
            label = display or Path(target.split("#", 1)[0]).stem
            if shared.CJK_RE.search(label):
                label = self.request(label)
            return protect(f"[[{target}|{label}]]")

        text = shared.WIKILINK_RE.sub(wiki, text)

        def markdown_link(match: re.Match[str]) -> str:
            if match.group(0).startswith("!"):
                return ""
            label, target = re.match(r"\[([^\]]*)\]\(([^)]*)\)", match.group(0)).groups()
            if shared.CJK_RE.search(label):
                label = self.request(label)
            return protect(f"[{label}]({target})")

        text = re.sub(r"!?\[[^\]]*\]\([^)]*\)", markdown_link, text)
        text = re.sub(r"https?://\S+", lambda match: protect(match.group(0)), text)
        text = text.replace("|", protect("|")) if "|" in text else text
        text = re.sub(
            r"^(\s*(?:[-*+]\s+|\d+[.)]\s+))",
            lambda match: protect(match.group(1)),
            text,
            flags=re.M,
        )
        for source, target in sorted(GLOSSARY, key=lambda pair: -len(pair[0])):
            if source in text:
                text = text.replace(source, protect(target))
        text = re.sub(r"\d+(?:[,.]\d+)*", lambda match: protect(match.group(0)), text)
        return text, replacements

    @staticmethod
    def restore(text: str, replacements: dict[str, str]) -> str:
        for key, value in reversed(replacements.items()):
            text = re.sub(re.escape(key), lambda _: value, text, flags=re.I)

        def restore_fuzzy(match: re.Match[str]) -> str:
            candidate = match.group(0).upper()
            ranked = sorted(
                (
                    difflib.SequenceMatcher(None, candidate, key).ratio(),
                    key,
                )
                for key in replacements
            )
            best_score, best_key = ranked[-1]
            second_score = ranked[-2][0] if len(ranked) > 1 else 0.0
            if best_score < 0.78 or best_score - second_score < 0.04:
                raise RuntimeError(
                    f"ambiguous placeholder {candidate}: "
                    f"best={best_key} score={best_score:.3f} margin="
                    f"{best_score - second_score:.3f}"
                )
            return replacements[best_key]

        text = re.sub(r"\bQZ[A-Z]{8,18}VQ\b", restore_fuzzy, text, flags=re.I)
        if re.search(r"QZ[A-Z]{12}VQ", text, flags=re.I):
            raise RuntimeError("unrestored placeholder: " + repr(text[:800]))
        return text.strip()

    def translate(self, unit: dict) -> str:
        text, replacements = self.prepare(unit)
        english = self.restore(self.request(text), replacements)
        if unit["kind"] == "heading":
            source_number = re.match(r"^(\d+(?:\.\d+)*\.?)\s*", unit["text"])
            if source_number:
                english = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", english)
                english = f"{source_number.group(1)} {english.strip()}"
        return english


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path
    baseline = Path("tmp/bilingual_baseline/classroom") / path
    if not baseline.exists():
        raise SystemExit(f"missing baseline: {baseline}")
    original = baseline.read_text()
    if shared.START in path.read_text():
        raise SystemExit(f"already bilingual: {path}")
    units = shared.collect_units(original.splitlines())
    translator = Translator()
    translations: dict[int, str] = {}
    for unit in units:
        english = translator.translate(unit)
        translations[unit["id"]] = english
    content = shared.make_content(original, units, translations)
    print(shared.patch_for(path, content), end="")


if __name__ == "__main__":
    main()
