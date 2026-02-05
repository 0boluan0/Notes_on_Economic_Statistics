#!/usr/bin/env python3
import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="Pydantic serializer warnings*")
warnings.filterwarnings("ignore", message="Error fetching version info*")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")


@dataclass
class OcrResult:
    kind: str  # "text" or "latex"
    value: str
    conf: float


class OcrEngine:
    def __init__(self, lang: str, min_conf: int) -> None:
        self.lang = lang
        self.min_conf = min_conf
        self._latex_ocr = None
        self._latex_disabled = False
        self._latex_warned = False

    def ocr_text(self, img) -> Tuple[str, float]:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            img,
            lang=self.lang,
            output_type=Output.DICT,
            config="--oem 3 --psm 6",
        )
        confs: List[int] = []
        for c in data.get("conf", []):
            try:
                ic = int(c)
            except Exception:
                continue
            if ic >= 0:
                confs.append(ic)
        conf = sum(confs) / len(confs) if confs else 0.0
        text = pytesseract.image_to_string(
            img, lang=self.lang, config="--oem 3 --psm 6"
        )
        text = normalize_text(text)
        return text, conf

    def ocr_latex(self, img) -> str:
        if self._latex_disabled:
            return ""
        if self._latex_ocr is None:
            try:
                from pix2tex.cli import LatexOCR

                self._latex_ocr = LatexOCR()
            except Exception as exc:
                self._latex_disabled = True
                if not self._latex_warned:
                    print(f"LatexOCR unavailable: {exc}", file=sys.stderr)
                    self._latex_warned = True
                return ""
        try:
            latex = self._latex_ocr(img)
        except Exception as exc:
            if not self._latex_warned:
                print(f"LatexOCR failed: {exc}", file=sys.stderr)
                self._latex_warned = True
            return ""
        return normalize_latex(latex)

    def choose_result(self, img) -> Optional[OcrResult]:
        text, conf = self.ocr_text(img)
        if text and conf >= self.min_conf and len(text) >= 5:
            return OcrResult(kind="text", value=text, conf=conf)
        latex = self.ocr_latex(img)
        if latex and is_mathy(latex):
            return OcrResult(kind="latex", value=latex, conf=conf)
        return None


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_latex(latex: str) -> str:
    latex = latex.strip()
    if latex.startswith("$$") and latex.endswith("$$"):
        latex = latex[2:-2].strip()
    if latex.startswith("$") and latex.endswith("$"):
        latex = latex[1:-1].strip()
    if latex.startswith("\\[") and latex.endswith("\\]"):
        latex = latex[2:-2].strip()
    latex = re.sub(r"\s+", " ", latex)
    return latex.strip()


def is_mathy(latex: str) -> bool:
    tokens = ["\\", "^", "_", "\\frac", "\\sum", "\\int", "\\sqrt", "=", "\\alpha", "\\beta"]
    return any(tok in latex for tok in tokens)


def split_prefix(line: str) -> Tuple[str, str, str]:
    nl = "\n" if line.endswith("\n") else ""
    raw = line[:-1] if nl else line
    indent = re.match(r"^\s*", raw).group(0)
    rest = raw[len(indent) :]
    prefix = indent
    while rest.startswith(">"):
        prefix += ">"
        rest = rest[1:]
        if rest.startswith(" "):
            prefix += " "
            rest = rest[1:]
    m = re.match(r"^([*+-])\s+(.*)$", rest)
    if m:
        prefix += m.group(1) + " "
        rest = m.group(2)
    else:
        m = re.match(r"^(\d+)\.\s+(.*)$", rest)
        if m:
            prefix += m.group(1) + ". "
            rest = m.group(2)
    return prefix, rest, nl


def parse_target(inner: str) -> str:
    target = inner.split("|", 1)[0].strip()
    target = target.split("#", 1)[0].strip()
    return target


def resolve_image_path(
    target: str,
    vault_root: Path,
    attachments_dir: Path,
    attachment_index: Dict[str, List[Path]],
) -> Optional[Path]:
    if "/" in target or "\\" in target:
        candidate = (vault_root / target).resolve()
        return candidate if candidate.exists() else None
    direct = attachments_dir / target
    if direct.exists():
        return direct
    if target in attachment_index:
        paths = attachment_index[target]
        if len(paths) == 1:
            return paths[0]
        return None
    candidate = (vault_root / target).resolve()
    return candidate if candidate.exists() else None


def is_struck(line: str, start: int, end: int) -> bool:
    return line[:start].endswith("~~") and line[end:].startswith("~~")


def preprocess_image(path: Path):
    from PIL import Image, ImageEnhance, ImageOps

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    return gray


def build_attachment_index(attachments_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    if not attachments_dir.exists():
        return index
    for path in attachments_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def process_line(
    line: str,
    line_no: int,
    file_path: Path,
    vault_root: Path,
    attachments_dir: Path,
    attachment_index: Dict[str, List[Path]],
    ocr: OcrEngine,
    cache: Dict[Path, Optional[OcrResult]],
    stats: Dict[str, int],
    dry_run: bool,
) -> Tuple[List[str], bool]:
    if "已转文字" in line:
        stats["skipped_existing"] += 1
        return [line], False

    prefix, rest, nl = split_prefix(line)
    if not EMBED_RE.search(rest):
        return [line], False

    matches = list(EMBED_RE.finditer(rest))
    standalone = (
        len(matches) == 1
        and rest.strip() == matches[0].group(0)
        and not is_struck(rest, matches[0].start(), matches[0].end())
    )

    if standalone:
        embed = matches[0].group(0)
        inner = matches[0].group(1)
        target = parse_target(inner)
        ext = Path(target).suffix.lower()
        if ext not in IMAGE_EXTS:
            stats["skipped_not_image"] += 1
            return [line], False
        image_path = resolve_image_path(target, vault_root, attachments_dir, attachment_index)
        if image_path is None or not image_path.exists():
            stats["skipped_missing"] += 1
            return [line], False
        if image_path in cache:
            result = cache[image_path]
        else:
            try:
                img = preprocess_image(image_path)
            except Exception:
                cache[image_path] = None
                stats["skipped_error"] += 1
                return [line], False
            result = ocr.choose_result(img)
            cache[image_path] = result
        if result is None:
            stats["skipped_low_conf"] += 1
            return [line], False
        if result.kind == "latex":
            replacement = f"$$ {result.value} $$"
        else:
            replacement = result.value
        new_line = f"{prefix}{replacement}{nl}"
        mark_line = f"{prefix}~~{embed}~~ (已转文字){nl or '\n'}"
        stats["replaced"] += 1
        log_action(file_path, line_no, embed, result.kind, result.conf, dry_run)
        return [new_line, mark_line], True

    # Inline replacements
    out = []
    last = 0
    changed = False
    for match in matches:
        start, end = match.start(), match.end()
        embed = match.group(0)
        inner = match.group(1)
        out.append(rest[last:start])
        last = end

        if is_struck(rest, start, end):
            out.append(embed)
            continue

        target = parse_target(inner)
        ext = Path(target).suffix.lower()
        if ext not in IMAGE_EXTS:
            out.append(embed)
            stats["skipped_not_image"] += 1
            continue

        image_path = resolve_image_path(target, vault_root, attachments_dir, attachment_index)
        if image_path is None or not image_path.exists():
            out.append(embed)
            stats["skipped_missing"] += 1
            continue

        if image_path in cache:
            result = cache[image_path]
        else:
            try:
                img = preprocess_image(image_path)
            except Exception:
                cache[image_path] = None
                stats["skipped_error"] += 1
                out.append(embed)
                continue
            result = ocr.choose_result(img)
            cache[image_path] = result
        if result is None:
            out.append(embed)
            stats["skipped_low_conf"] += 1
            continue

        if result.kind == "latex":
            replacement = f"${result.value}$"
        else:
            replacement = result.value
        out.append(f"{replacement} ~~{embed}~~ (已转文字)")
        stats["replaced"] += 1
        changed = True
        log_action(file_path, line_no, embed, result.kind, result.conf, dry_run)

    out.append(rest[last:])
    new_line = prefix + "".join(out) + nl
    return [new_line], changed


def log_action(file_path: Path, line_no: int, embed: str, kind: str, conf: float, dry_run: bool) -> None:
    mode = "DRY" if dry_run else "WRITE"
    print(f"[{mode}] {file_path}:{line_no} {kind} conf={conf:.1f} {embed}")


def process_file(
    path: Path,
    vault_root: Path,
    attachments_dir: Path,
    attachment_index: Dict[str, List[Path]],
    ocr: OcrEngine,
    cache: Dict[Path, Optional[OcrResult]],
    stats: Dict[str, int],
    dry_run: bool,
) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return False

    lines = content.splitlines(keepends=True)
    in_frontmatter = False
    in_code = False
    changed = False
    new_lines: List[str] = []

    for idx, line in enumerate(lines, start=1):
        if idx == 1 and line.strip() == "---":
            in_frontmatter = True
            new_lines.append(line)
            continue
        if in_frontmatter:
            new_lines.append(line)
            if line.strip() == "---":
                in_frontmatter = False
            continue
        if re.match(r"^\s*(```|~~~)", line):
            in_code = not in_code
            new_lines.append(line)
            continue
        if in_code:
            new_lines.append(line)
            continue
        if "![[" not in line:
            new_lines.append(line)
            continue
        processed, line_changed = process_line(
            line,
            idx,
            path,
            vault_root,
            attachments_dir,
            attachment_index,
            ocr,
            cache,
            stats,
            dry_run,
        )
        new_lines.extend(processed)
        if line_changed:
            changed = True

    if changed and not dry_run:
        path.write_text("".join(new_lines), encoding="utf-8")
    return changed


def iter_markdown_files(roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        for path in root.rglob("*.md"):
            if "/.obsidian/" in str(path):
                continue
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR replace image embeds in Obsidian notes")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--attachments", default="98_attachment")
    parser.add_argument("--lang", default="eng+chi_sim")
    parser.add_argument("--min-conf", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vault_root = Path.cwd().resolve()
    roots = [vault_root / r for r in args.roots]
    attachments_dir = (vault_root / args.attachments).resolve()

    for root in roots:
        if not root.exists():
            print(f"Root not found: {root}", file=sys.stderr)
            return 1

    attachment_index = build_attachment_index(attachments_dir)
    ocr = OcrEngine(lang=args.lang, min_conf=args.min_conf)
    cache: Dict[Path, Optional[OcrResult]] = {}
    stats = {
        "replaced": 0,
        "skipped_low_conf": 0,
        "skipped_missing": 0,
        "skipped_not_image": 0,
        "skipped_existing": 0,
        "skipped_error": 0,
    }

    files = iter_markdown_files(roots)
    changed_files = 0
    for path in files:
        if process_file(
            path,
            vault_root,
            attachments_dir,
            attachment_index,
            ocr,
            cache,
            stats,
            args.dry_run,
        ):
            changed_files += 1

    print("---")
    print(f"files_scanned: {len(files)}")
    print(f"files_changed: {changed_files}")
    for key, val in stats.items():
        print(f"{key}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
