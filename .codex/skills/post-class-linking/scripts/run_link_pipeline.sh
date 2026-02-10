#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_link_pipeline.sh preview [--full|--incremental] [--no-inject-backlink-panel]
  run_link_pipeline.sh apply   [--full|--incremental] [--no-inject-backlink-panel]

Notes:
- preview: dry-run only
- apply:   write files and update reports/linking/.state.json
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ACTION="$1"
shift

if [[ "$ACTION" != "preview" && "$ACTION" != "apply" ]]; then
  usage
  exit 1
fi

FORCE_MODE=""
INJECT_PANEL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      FORCE_MODE="full"
      ;;
    --incremental)
      FORCE_MODE="incremental"
      ;;
    --no-inject-backlink-panel)
      INJECT_PANEL=0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

STATE_DIR="$ROOT_DIR/reports/linking"
STATE_FILE="$STATE_DIR/.state.json"
CHANGED_FILE="$STATE_DIR/.changed_files.txt"
mkdir -p "$STATE_DIR"

CURRENT_HEAD="$(git rev-parse HEAD)"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"

update_state() {
  local mode="$1"
  python3 - "$STATE_FILE" "$CURRENT_HEAD" "$mode" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_file = Path(sys.argv[1])
commit = sys.argv[2]
mode = sys.argv[3]
state = {
    "last_processed_commit": commit,
    "last_run_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "last_mode": mode,
}
state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"state updated: {state_file}")
PY
}

read_last_commit() {
  python3 - "$STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

print(data.get("last_processed_commit", ""))
PY
}

MODE=""
if [[ -n "$FORCE_MODE" ]]; then
  MODE="$FORCE_MODE"
elif [[ -f "$STATE_FILE" ]]; then
  MODE="incremental"
else
  MODE="full"
fi

EXTRA_ARGS=()
if [[ $INJECT_PANEL -eq 0 ]]; then
  EXTRA_ARGS+=(--no-inject-backlink-panel)
fi

if [[ "$MODE" == "incremental" ]]; then
  LAST_COMMIT="$(read_last_commit)"
  if [[ -z "$LAST_COMMIT" ]]; then
    echo "state file missing/invalid commit. please run --full first."
    exit 2
  fi

  if ! git cat-file -e "${LAST_COMMIT}^{commit}" 2>/dev/null; then
    echo "invalid baseline commit in state: $LAST_COMMIT"
    echo "please run: run_link_pipeline.sh preview --full"
    exit 3
  fi

  tmp_changed="$(mktemp)"
  git diff --name-only "${LAST_COMMIT}..${CURRENT_HEAD}" -- 01_Math 02_Economy 03_Computer_Science >> "$tmp_changed" || true
  git diff --name-only --cached -- 01_Math 02_Economy 03_Computer_Science >> "$tmp_changed" || true
  git diff --name-only -- 01_Math 02_Economy 03_Computer_Science >> "$tmp_changed" || true
  rg '\.md$' "$tmp_changed" | sort -u > "$CHANGED_FILE" || true
  rm -f "$tmp_changed"

  CHANGED_COUNT="$(wc -l < "$CHANGED_FILE" | tr -d ' ')"
  echo "incremental baseline: $LAST_COMMIT"
  echo "changed markdown files: $CHANGED_COUNT"

  if [[ "$CHANGED_COUNT" == "0" ]]; then
    echo "no course markdown changes detected."
    if [[ "$ACTION" == "apply" ]]; then
      update_state "incremental"
    fi
    exit 0
  fi

  EXTRA_ARGS+=(--mode incremental --changed-files-file "$CHANGED_FILE")
else
  EXTRA_ARGS+=(--mode full)
fi

REPORT_PATH="$STATE_DIR/link_${MODE}_${ACTION}_${TIMESTAMP}.json"
CMD=(python3 "$ROOT_DIR/link_knowledge.py" "${EXTRA_ARGS[@]}" --report-path "$REPORT_PATH")

if [[ "$ACTION" == "preview" ]]; then
  CMD+=(--dry-run)
fi

echo "running: ${CMD[*]}"
"${CMD[@]}"

if [[ "$ACTION" == "apply" ]]; then
  update_state "$MODE"
fi

echo "report saved: $REPORT_PATH"
