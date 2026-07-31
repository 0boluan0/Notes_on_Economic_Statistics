#!/usr/bin/env python3
"""Apply reviewed-structure bilingual drafts to a list of classroom notes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import bilingualize_classroom as bilingual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--model", default="caiyun-translate")
    args = parser.parse_args()

    for path in args.paths:
        baseline = Path("tmp/bilingual_baseline/classroom") / path
        if not baseline.exists():
            raise SystemExit(f"missing baseline: {baseline}")
        current = path.read_text()
        original = baseline.read_text()
        if bilingual.START in current:
            print(f"SKIP {path}", flush=True)
            continue
        if current.rstrip("\n") != original.rstrip("\n"):
            raise SystemExit(f"current file differs from baseline: {path}")
        original = current
        units = bilingual.collect_units(original.splitlines())
        try:
            translations = bilingual.request_translations(units, args.model)
        except Exception as exc:
            print(f"FAIL {path}: {exc}", flush=True)
            continue
        if path.name == "06_经济增长理论.md" and 8 in translations:
            translations[8] = translations[8].replace("常数", "constant")
        content = bilingual.make_content(original, units, translations)
        patch = bilingual.patch_for(path, content)
        result = subprocess.run(
            ["apply_patch"],
            input=patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"apply_patch failed for {path}: {result.stdout}{result.stderr}"
            )
        print(f"DONE {path} units={len(units)}", flush=True)


if __name__ == "__main__":
    main()
