#!/usr/bin/env python3
"""
pull_s3_creds_from_vm.py — Copy the DGT LiDAR S3/MinIO credentials from the
JupyterHub VM into the local project `.env`, WITHOUT ever printing the secrets.

Run this yourself in your terminal (it keeps secrets off any shared transcript):

    python scripts/pull_s3_creds_from_vm.py

It executes a tiny snippet on the VM via scripts/jupyterhub_control.py, captures
the AWS_*1/AWS_*2 env vars, appends/updates them in `.env`, and prints only a
masked confirmation (e.g. AKIA…last4).  Existing non-AWS lines in `.env` are kept.
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ENV_PATH = ROOT / ".env"

KEYS = [
    "AWS_ENDPOINT_URL2", "AWS_ACCESS_KEY_ID2", "AWS_SECRET_ACCESS_KEY2",
    "AWS_ENDPOINT_URL1", "AWS_ACCESS_KEY_ID1", "AWS_SECRET_ACCESS_KEY1",
    "AWS_S3_BUCKET",
]

VM_SNIPPET = (
    "import os\n"
    "for k in " + repr(KEYS) + ":\n"
    "    print('CRED::%s=%s' % (k, os.environ.get(k, '')))\n"
)


def mask(v: str) -> str:
    if not v:
        return "(empty)"
    if v.startswith("http"):
        return v  # endpoints aren't secret
    return (v[:4] + "…" + v[-4:]) if len(v) > 8 else "•••"


def main() -> None:
    import jupyterhub_control as jhc  # noqa: E402

    # Run on the VM, capturing outputs WITHOUT echoing them to the terminal.
    buf = io.StringIO()
    with redirect_stdout(buf):
        session_id, kernel_id = jhc._create_session()
        try:
            outputs = jhc._execute_on_kernel(kernel_id, VM_SNIPPET)
        finally:
            jhc._delete_session(session_id)
    raw = "\n".join(outputs) + "\n" + buf.getvalue()

    creds: dict[str, str] = {}
    for m in re.finditer(r"CRED::([A-Z0-9_]+)=(.*)", raw):
        k, v = m.group(1), m.group(2).strip()
        if k in KEYS and v:
            creds[k] = v

    if not creds:
        print("ERROR: no credentials returned from the VM. Is the JupyterHub up?")
        sys.exit(1)

    # Merge into .env, replacing any existing AWS_* lines, keeping the rest.
    existing = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    kept = [ln for ln in existing if not any(ln.startswith(k + "=") for k in KEYS)]
    new_lines = kept + [f"{k}={creds[k]}" for k in KEYS if k in creds]
    ENV_PATH.write_text("\n".join(new_lines) + "\n")

    print(f"✓ wrote {len(creds)} S3 vars to {ENV_PATH} (secrets not displayed):")
    for k in KEYS:
        if k in creds:
            print(f"    {k} = {mask(creds[k])}")


if __name__ == "__main__":
    main()
