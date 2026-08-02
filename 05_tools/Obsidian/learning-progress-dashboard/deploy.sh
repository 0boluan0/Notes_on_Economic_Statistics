#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_ROOT="$(cd "$SOURCE_DIR/../../.." && pwd)"
DESTINATION="$VAULT_ROOT/.obsidian/plugins/learning-progress-dashboard"

node --check "$SOURCE_DIR/main.js"
node "$SOURCE_DIR/self-check.js"
mkdir -p "$DESTINATION"
install -m 0644 "$SOURCE_DIR/main.js" "$DESTINATION/main.js"
install -m 0644 "$SOURCE_DIR/styles.css" "$DESTINATION/styles.css"
install -m 0644 "$SOURCE_DIR/manifest.json" "$DESTINATION/manifest.json"

cmp "$SOURCE_DIR/main.js" "$DESTINATION/main.js"
cmp "$SOURCE_DIR/styles.css" "$DESTINATION/styles.css"
cmp "$SOURCE_DIR/manifest.json" "$DESTINATION/manifest.json"
echo "learning-progress-dashboard deployed"
