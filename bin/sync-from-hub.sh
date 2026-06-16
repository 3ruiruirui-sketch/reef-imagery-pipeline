#!/usr/bin/env bash
#
# sync-from-hub.sh — pull code/data from the DGT JupyterHub back to local
#
# Usage:
#   bin/sync-from-hub.sh
#   bin/sync-from-hub.sh --include outputs/   # pull back generated outputs too
#
# Useful for retrieving:
#   - models retrained on the hub
#   - orchestrator reports (orchestrator_report.json)
#   - drift HTML reports
#   - any data the hub produced

set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="dgt-jhub"
REMOTE_PATH="~/reef-imagery-pipeline/"

# Default: only pull specific artefacts
EXCLUDES=(
  --exclude='.venv/'
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='*.pyc'
  --exclude='.env'
  --exclude='.DS_Store'
)

# Only pull these (small, valuable artefacts)
INCLUDES=(
  --include='models/'
  --include='models/**'
  --include='outputs/'
  --include='outputs/**'
  --include='*.json'
  --include='*.csv'
  --include='*.html'
  --include='*.tif'
  --include='*.geojson'
  --include='*.ipynb'
  --exclude='*'
)

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN="--dry-run"
  echo "DRY RUN: no files will be transferred"
fi

if ! command -v rsync >/dev/null; then
  echo "ERROR: rsync not installed." >&2
  exit 1
fi

if ! ssh -q -o ConnectTimeout=10 "${REMOTE_HOST}" true 2>/dev/null; then
  echo "ERROR: cannot reach ${REMOTE_HOST}." >&2
  exit 1
fi

echo "Pulling from ${REMOTE_HOST}:${REMOTE_PATH} to ${LOCAL_DIR}"
rsync -avz --delete $DRY_RUN "${INCLUDES[@]}" "${EXCLUDES[@]}" \
  -e "ssh -p 8022" \
  "${REMOTE_HOST}:${REMOTE_PATH}/" \
  "${LOCAL_DIR}/"
