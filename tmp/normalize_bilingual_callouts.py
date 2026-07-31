from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
START = re.compile(r"^\s*(?:>\s*)*<!-- bilingual-en:start -->\s*$")
END = re.compile(r"^\s*(?:>\s*)*<!-- bilingual-en:end -->\s*$")
CALLOUT = re.compile(r"^(\s*(?:>\s*)+)\[![^]]+\].*$")
REDUNDANT_TITLE = re.compile(
    r"^\*\*\s*(?:self[- ]?(?:test|check) answers?|answer|quick recovery|sources? for this section)\s*\*\*$",
    re.IGNORECASE,
)


def depth(line: str) -> int:
    return len(re.match(r"^\s*((?:>\s*)*)", line).group(1).replace(" ", ""))


def strip_english(text: str) -> str:
    output = []
    inside = False
    for line in text.splitlines(keepends=True):
        if START.match(line):
            inside = True
            continue
        if END.match(line):
            inside = False
            continue
        if not inside:
            output.append(line)
    assert not inside
    return "".join(output)


def normalize(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    output = []
    changed = 0
    i = 0
    while i < len(lines):
        if not START.match(lines[i]):
            output.append(lines[i])
            i += 1
            continue

        j = i + 1
        while j < len(lines) and not END.match(lines[j]):
            j += 1
        assert j < len(lines), "Unclosed bilingual block"

        body = lines[i + 1 : j]
        first = next((k for k, line in enumerate(body) if line.strip()), None)
        previous = next((line for line in reversed(output) if line.strip()), "")
        if first is None or not previous.lstrip().startswith(">"):
            output.extend(lines[i : j + 1])
            i = j + 1
            continue

        declaration = CALLOUT.match(body[first])
        title_prefix = re.match(r"^(\s*(?:>\s*)+)(.*)$", body[first])
        redundant_title = bool(
            title_prefix and REDUNDANT_TITLE.match(title_prefix.group(2).strip())
        )
        if not declaration and not redundant_title:
            output.extend(lines[i : j + 1])
            i = j + 1
            continue

        prefix = (declaration or title_prefix).group(1)
        quote_depth = depth(prefix)
        marker_prefix = ">" * quote_depth
        output.append(f"{marker_prefix} <!-- bilingual-en:start -->\n")
        for k, line in enumerate(body):
            if k == first:
                continue
            if not line.strip():
                output.append(f"{marker_prefix}\n")
            elif depth(line) < quote_depth:
                output.append(f"{marker_prefix} {line.lstrip()}")
            else:
                output.append(line)
        output.append(f"{marker_prefix} <!-- bilingual-en:end -->\n")
        changed += 1
        i = j + 1

    result = "".join(output)
    assert strip_english(result) == strip_english(text)
    return result, changed


def main() -> None:
    write = sys.argv[1:] == ["--write"]
    total = 0
    files = 0
    for path in sorted((ROOT / "00_Knowledge").glob("**/*.md")):
        original = path.read_text()
        updated, changed = normalize(original)
        if changed:
            files += 1
            total += changed
            if write:
                path.write_text(updated)
    print(f"files={files} callouts={total} mode={'write' if write else 'check'}")


if __name__ == "__main__":
    main()
