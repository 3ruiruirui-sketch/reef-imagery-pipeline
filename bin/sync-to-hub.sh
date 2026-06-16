#!/usr/bin/env bash
#
# sync-to-hub.sh — push local code to the DGT JupyterHub
#
# Usage:
#   bin/sync-to-hub.sh
#   bin/sync-to-hub.sh --dry-run
#
# Excludes (kept local-only):
#   .venv/        — virtual env (rebuild on hub)
#   __pycache__/  — Python bytecode
#   .pytest_cache/
#   *.pyc
#   .env          — secrets, never sync
#   outputs/      — large generated artefacts
#   *.tif, *.tiff — large imagery
#   data/cache/   — large downloaded tiles
#   .DS_Store
#
# Requires: rsync, ssh (configured in ~/.ssh/config with a `dgt-jhub` host)

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="dgt-jhub"                  # alias in ~/.ssh/config
REMOTE_PATH="~/reef-imagery-pipeline/"

EXCLUDES=(
  --exclude='.venv/'
  --exclude='__pycache__/'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='*.pyc'
  --exclude='*.pyo'
  --exclude='.env'
  --exclude='.env.local'
  --exclude='outputs/'
  --exclude='*.tif'
  --exclude='*.tiff'
  --exclude='data/cache/'
  --exclude='dashboard/tiles/'
  --exclude='dashboard/full_dpi_cache/'
  --exclude='.DS_Store'
  --exclude='*.egg-info/'
  --exclude='.ipynb_checkpoints/'
)

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN="--dry-run"
  echo "DRY RUN: no files will be transferred"
fi

# ── Pre-flight ─────────────────────────────────────────────────────────────
if ! command -v rsync >/dev/null; then
  echo "ERROR: rsync not installed. Install with: brew install rsync" >&2
  exit 1
fi

if ! ssh -q -o ConnectTimeout=10 "${REMOTE_HOST}" true 2>/dev/null; then
  echo "ERROR: cannot reach ${REMOTE_HOST}. Check ~/.ssh/config and that the" >&2
  echo "       public key (~/.ssh/id_ed25519.pub) is added at" >&2
  echo "       https://dgt-jupyterhub.d.acnca.pt → Hub Control Panel → SSH Keys" >&2
  exit 1
fi

# ── Sync ───────────────────────────────────────────────────────────────────
echo "Local : ${LOCAL_DIR}"
echo "Remote: ${REMOTE_HOST}:${REMOTE_PATH}"
echo

rsync -avz --delete $DRY_RUN "${EXCLUDES[@]}" \
  -e "ssh -p 8022" \
  "${LOCAL_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_PATH}/"

# ── Post-sync hint ──────────────────────────────────────────────────────────
cat <<'EOF'

Synced. To pick up Python changes on the hub:
  ssh dgt-jhub 'cd reef-imagery-pipeline && source .venv/bin/activate && pip install -e . --quiet'

To pull a JupyterHub-trained model back to local:
  bin/sync-from-hub.sh
EOF
