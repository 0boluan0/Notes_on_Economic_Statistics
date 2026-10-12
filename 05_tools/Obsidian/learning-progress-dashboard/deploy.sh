#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_ROOT="$(cd "$SOURCE_DIR/../../.." && pwd)"
DESTINATION="$VAULT_ROOT/.obsidian/plugins/learning-progress-dashboard"

node --check "$SOURCE_DIR/main.js"
node --check "$SOURCE_DIR/steward-view.js"
node "$SOURCE_DIR/self-check.js"
PYTHON_BIN="$(command -v python3)"
"$PYTHON_BIN" -B -m unittest discover -s "$VAULT_ROOT/05_tools/StudentOS/steward" -p 'test_*.py'
STUDENT_OS_BUNDLE="$(mktemp -t student-os-dashboard)"
trap 'rm -f "$STUDENT_OS_BUNDLE"' EXIT
node "$SOURCE_DIR/build.js" > "$STUDENT_OS_BUNDLE"
node --check < "$STUDENT_OS_BUNDLE"
mkdir -p "$DESTINATION"
install -m 0644 "$STUDENT_OS_BUNDLE" "$DESTINATION/main.js"
install -m 0644 "$SOURCE_DIR/styles.css" "$DESTINATION/styles.css"
install -m 0644 "$SOURCE_DIR/manifest.json" "$DESTINATION/manifest.json"

cmp "$STUDENT_OS_BUNDLE" "$DESTINATION/main.js"
cmp "$SOURCE_DIR/styles.css" "$DESTINATION/styles.css"
cmp "$SOURCE_DIR/manifest.json" "$DESTINATION/manifest.json"
echo "learning-progress-dashboard deployed"
