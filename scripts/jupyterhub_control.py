#!/usr/bin/env python3
"""
jupyterhub_control.py — Remote JupyterHub/Jupyter CLI controller for macOS.

Target server : https://dgt-jupyterhub.d.acnca.pt
User          : jsoares
User path     : /user/jsoares/

Dependencies  : pip install requests websocket-client   (already in .venv)

Usage examples
--------------
  python scripts/jupyterhub_control.py whoami
  python scripts/jupyterhub_control.py ls
  python scripts/jupyterhub_control.py ls --path reef-imagery-pipeline/notebooks
  python scripts/jupyterhub_control.py run-code "print('hello from JupyterHub')"
  python scripts/jupyterhub_control.py run-script scripts/reef_bathy_module.py
  python scripts/jupyterhub_control.py upload-notebook notebooks/my.ipynb
  python scripts/jupyterhub_control.py s3-ls
  python scripts/jupyterhub_control.py s3-ls --backend mdt --prefix MDT2m/ --max 10
  python scripts/jupyterhub_control.py s3-stream-test
  python scripts/jupyterhub_control.py sync-project
  python scripts/jupyterhub_control.py clone-repo https://github.com/user/repo

NOTE ON AUTH
------------
The JupyterHub WebSocket (/api/kernels/{id}/channels) requires BOTH:
  1. Authorization: token <TOKEN>   header  (HTTP API calls)
  2. Cookie: _xsrf=...              cookie  (XSRF protection on the Tornado server)
The XSRF cookie is obtained by visiting /lab once with the token; this script
does it automatically via a session.get() before opening any WebSocket.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import threading
import uuid
from pathlib import Path

import requests
import websocket  # pip install websocket-client


# ---------------------------------------------------------------------------
# Configuration — loaded from .env at project root
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Minimal .env loader (no dependency): project-root .env → os.environ."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

BASE_URL  = os.environ.get("JUPYTERHUB_BASE_URL",  "https://dgt-jupyterhub.d.acnca.pt")
USERNAME  = os.environ.get("JUPYTERHUB_USERNAME",   "jsoares")
TOKEN     = os.environ.get("JUPYTERHUB_API_TOKEN",  "")
HUB_API   = os.environ.get("JUPYTERHUB_HUB_API_URL", f"{BASE_URL}/hub/api")

if not TOKEN:
    sys.exit(
        "ERROR: JupyterHub token not set.\n"
        "Add JUPYTERHUB_API_TOKEN=<token> to the project .env file (gitignored).\n"
        f"Get/rotate a token at {BASE_URL}/hub/token"
    )

USER_BASE = f"{BASE_URL}/user/{USERNAME}"
BASE_WS   = USER_BASE.replace("https://", "wss://").replace("http://", "ws://")

HTTP_HEADERS   = {"Authorization": f"token {TOKEN}", "Content-Type": "application/json"}
HTTP_TIMEOUT   = 30
KERNEL_TIMEOUT = int(os.environ.get("JHUB_EXEC_TIMEOUT", "180"))

# Project root (one level above scripts/)
PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Session with XSRF cookie (required for WebSocket auth)
# ---------------------------------------------------------------------------

_http_session: requests.Session | None = None
_xsrf_token: str = ""


def _get_http_session() -> tuple[requests.Session, str]:
    """Return a requests.Session with valid XSRF cookie for WebSocket auth."""
    global _http_session, _xsrf_token
    if _http_session and _xsrf_token:
        return _http_session, _xsrf_token

    s = requests.Session()
    s.headers.update({"Authorization": f"token {TOKEN}"})
    s.verify = False  # DGT uses self-signed / Caddy cert
    # Suppress InsecureRequestWarning
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Fetch /lab to obtain the _xsrf cookie from Tornado
    r = s.get(f"{USER_BASE}/lab", allow_redirects=True, timeout=HTTP_TIMEOUT)
    if r.status_code not in (200, 302):
        print(f"WARNING: /lab returned {r.status_code} — XSRF cookie may be missing.", file=sys.stderr)

    _xsrf_token = s.cookies.get("_xsrf", "")
    _http_session = s
    return _http_session, _xsrf_token


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, **kwargs) -> requests.Response:
    sess, _ = _get_http_session()
    r = sess.get(url, timeout=HTTP_TIMEOUT, **kwargs)
    _raise(r)
    return r


def _post(url: str, payload: dict | None = None, **kwargs) -> requests.Response:
    sess, xsrf = _get_http_session()
    extra_headers = {"X-XSRFToken": xsrf} if xsrf else {}
    r = sess.post(url, json=payload or {}, headers=extra_headers,
                  timeout=HTTP_TIMEOUT, **kwargs)
    _raise(r)
    return r


def _put(url: str, payload: dict) -> requests.Response:
    sess, xsrf = _get_http_session()
    extra_headers = {"X-XSRFToken": xsrf} if xsrf else {}
    r = sess.put(url, json=payload, headers=extra_headers, timeout=HTTP_TIMEOUT)
    _raise(r)
    return r


def _delete(url: str) -> requests.Response:
    sess, xsrf = _get_http_session()
    extra_headers = {"X-XSRFToken": xsrf} if xsrf else {}
    r = sess.delete(url, headers=extra_headers, timeout=HTTP_TIMEOUT)
    _raise(r)
    return r


def _raise(r: requests.Response) -> None:
    if r.status_code == 401:
        print("ERROR 401 — Unauthorized. Check your JUPYTERHUB_API_TOKEN.", file=sys.stderr)
        sys.exit(1)
    if r.status_code == 403:
        print(f"ERROR 403 — Forbidden: {r.url}", file=sys.stderr)
        sys.exit(1)
    if r.status_code == 404:
        print(f"ERROR 404 — Not found: {r.url}", file=sys.stderr)
        sys.exit(1)
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        print(f"ERROR {r.status_code} — {exc}\n{r.text[:400]}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Kernel / Session helpers
# ---------------------------------------------------------------------------

def _create_session(kernel_name: str = "python3") -> tuple[str, str]:
    """Create a new kernel session. Returns (session_id, kernel_id)."""
    url = f"{USER_BASE}/api/sessions"
    payload = {
        "kernel": {"name": kernel_name},
        "name": "",
        "path": f"cli-{uuid.uuid4().hex[:8]}",
        "type": "console",
    }
    data = _post(url, payload).json()
    session_id = data["id"]
    kernel_id  = data["kernel"]["id"]
    print(f"  kernel started  — {kernel_id}  [{kernel_name}]")
    return session_id, kernel_id


def _delete_session(session_id: str) -> None:
    """Delete a kernel session."""
    _delete(f"{USER_BASE}/api/sessions/{session_id}")
    print(f"  kernel session {session_id[:8]}... deleted.")


def _build_execute_request(code: str) -> dict:
    """Build a Jupyter protocol execute_request message."""
    msg_id = uuid.uuid4().hex
    return {
        "header": {
            "msg_id":   msg_id,
            "username": USERNAME,
            "session":  uuid.uuid4().hex,
            "msg_type": "execute_request",
            "version":  "5.3",
            "date":     "",
        },
        "parent_header": {},
        "metadata":      {},
        "content": {
            "code":             code,
            "silent":           False,
            "store_history":    True,
            "user_expressions": {},
            "allow_stdin":      False,
            "stop_on_error":    False,
        },
        "channel": "shell",
        "buffers": [],
    }


def _execute_on_kernel(kernel_id: str, code: str, verbose: bool = True) -> list[str]:
    """
    Open a WebSocket to the kernel, send execute_request, collect all output.
    Uses both the Authorization header AND the _xsrf cookie (required by Tornado).
    Returns a list of captured output strings.
    """
    _, xsrf = _get_http_session()
    # Build cookie string from the session
    sess, _ = _get_http_session()
    cookie_str = "; ".join(f"{k}={v}" for k, v in sess.cookies.items())

    ws_url = f"{BASE_WS}/api/kernels/{kernel_id}/channels"
    request = _build_execute_request(code)
    our_msg_id = request["header"]["msg_id"]

    outputs: list[str] = []
    done = threading.Event()

    def on_open(ws):
        ws.send(json.dumps(request))

    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        mtype  = msg.get("header", {}).get("msg_type", "")
        parent = msg.get("parent_header", {}).get("msg_id", "")

        # Only process replies to OUR message
        if parent != our_msg_id:
            return

        content = msg.get("content", {})

        if mtype == "stream":
            text = content.get("text", "")
            if verbose:
                print(text, end="", flush=True)
            outputs.append(text)

        elif mtype == "execute_result":
            text = content.get("data", {}).get("text/plain", "")
            if verbose:
                print(f"Out: {text}")
            outputs.append(text)

        elif mtype == "display_data":
            text = content.get("data", {}).get("text/plain", "")
            if verbose and text:
                print(f"Display: {text}")
            outputs.append(text)

        elif mtype == "error":
            tb = "\n".join(content.get("traceback", []))
            tb_clean = re.sub(r"\x1b\[[0-9;]*m", "", tb)
            evalue = content.get("evalue", "?")
            print(f"\nKERNEL ERROR [{content.get('ename')}]: {evalue}", file=sys.stderr)
            if tb_clean:
                print(tb_clean, file=sys.stderr)
            outputs.append(f"ERROR: {evalue}")
            done.set()

        elif mtype == "execute_reply":
            status = content.get("status", "?")
            if verbose:
                print(f"\n  [execute_reply: {status}]")
            done.set()

    def on_error(ws, error):
        print(f"\nWebSocket error: {error}", file=sys.stderr)
        done.set()

    def on_close(ws, code, msg):
        done.set()

    ws_app = websocket.WebSocketApp(
        ws_url,
        header=[
            f"Authorization: token {TOKEN}",
            f"Cookie: {cookie_str}",
            f"X-XSRFToken: {xsrf}",
        ],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    t = threading.Thread(
        target=lambda: ws_app.run_forever(
            sslopt={"cert_reqs": ssl.CERT_NONE},
            ping_interval=20,
        )
    )
    t.daemon = True
    t.start()

    if not done.wait(timeout=KERNEL_TIMEOUT):
        print(f"\nWARNING — no kernel reply within {KERNEL_TIMEOUT}s.", file=sys.stderr)
        ws_app.close()

    t.join(timeout=5)
    return outputs


def _run(code: str, verbose: bool = True) -> list[str]:
    """
    Convenience: create a kernel session, run code, delete session.
    Returns list of output strings.
    """
    session_id, kernel_id = _create_session()
    time.sleep(2)  # give kernel a moment to become ready
    try:
        return _execute_on_kernel(kernel_id, code, verbose=verbose)
    finally:
        _delete_session(session_id)


# ---------------------------------------------------------------------------
# Command: whoami
# ---------------------------------------------------------------------------

def cmd_whoami(_args) -> None:
    data = _get(f"{HUB_API}/users/{USERNAME}").json()
    print(f"  name         : {data.get('name')}")
    print(f"  admin        : {data.get('admin')}")
    print(f"  server       : {data.get('server') or '(not running)'}")
    print(f"  last_activity: {data.get('last_activity')}")
    print(f"  groups       : {', '.join(data.get('groups') or []) or '(none)'}")


# ---------------------------------------------------------------------------
# Command: ls
# ---------------------------------------------------------------------------

def cmd_ls(args) -> None:
    remote_path = (getattr(args, "path", "") or "").lstrip("/")
    data = _get(f"{USER_BASE}/api/contents/{remote_path}").json()
    content = sorted(data.get("content") or [],
                     key=lambda x: (x["type"] != "directory", x["name"].lower()))
    if not content:
        print("(empty)")
        return
    col = "{:<10}  {:<12}  {}"
    print(col.format("type", "size", "name"))
    print("-" * 60)
    for item in content:
        size_str = f"{item.get('size', 0):,}" if item["type"] != "directory" else "-"
        print(col.format(item["type"], size_str, item["name"]))


# ---------------------------------------------------------------------------
# Command: upload-notebook
# ---------------------------------------------------------------------------

def cmd_upload_notebook(args) -> None:
    local = Path(args.notebook)
    if not local.exists():
        sys.exit(f"ERROR — not found: {local}")
    remote = (getattr(args, "remote_path", None) or local.name).lstrip("/")
    payload = {
        "type":    "notebook",
        "format":  "json",
        "content": json.loads(local.read_text(encoding="utf-8")),
    }
    _put(f"{USER_BASE}/api/contents/{remote}", payload)
    print(f"Uploaded '{local}' → remote '{remote}'")


# ---------------------------------------------------------------------------
# Command: run-code
# ---------------------------------------------------------------------------

def cmd_run_code(args) -> None:
    code = args.code
    print(f"Running on {USER_BASE}:")
    print(f"  >>> {code[:120]}{'...' if len(code) > 120 else ''}\n")
    _run(code)


# ---------------------------------------------------------------------------
# Command: run-script
# Run a local Python script on the remote kernel
# ---------------------------------------------------------------------------

def cmd_run_script(args) -> None:
    local = Path(args.script)
    if not local.exists():
        sys.exit(f"ERROR — script not found: {local}")
    code = local.read_text(encoding="utf-8")
    print(f"Running '{local.name}' on {USER_BASE}...\n")
    _run(code)


# ---------------------------------------------------------------------------
# Command: s3-ls  — list objects in the DGT MinIO buckets
# ---------------------------------------------------------------------------

_S3_LS_CODE = """
import boto3, os, warnings
warnings.filterwarnings('ignore')

backend  = "{backend}"
prefix   = "{prefix}"
max_keys = {max_keys}

if backend in ("laz", "1"):
    ep  = os.getenv("AWS_ENDPOINT_URL1", "https://stor-001.a.acnca.pt:9000")
    key = os.getenv("AWS_ACCESS_KEY_ID1")
    sec = os.getenv("AWS_SECRET_ACCESS_KEY1")
    label = "stor-001 (LAZ)"
else:
    ep  = os.getenv("AWS_ENDPOINT_URL2", "https://stor-002.a.acnca.pt:9000")
    key = os.getenv("AWS_ACCESS_KEY_ID2")
    sec = os.getenv("AWS_SECRET_ACCESS_KEY2")
    label = "stor-002 (MDT/MDS/Ortos)"

bucket = os.getenv("AWS_S3_BUCKET", "lidar")
s3 = boto3.client("s3", endpoint_url=ep, aws_access_key_id=key,
                  aws_secret_access_key=sec, verify=False)

print(f"Backend : {{label}}")
print(f"Bucket  : {{bucket}}")
print(f"Prefix  : '{{prefix}}'")
print(f"Max keys: {{max_keys}}")
print()

r = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
objs = r.get("Contents", [])
print(f"Found {{len(objs)}} objects:")
for o in objs:
    size_mb = o["Size"] / 1e6
    print(f"  {{o['Key']:<70}}  {{size_mb:7.2f}} MB")
if r.get("IsTruncated"):
    print(f"  ... (more results, use --max to increase)")
"""

def cmd_s3_ls(args) -> None:
    backend  = getattr(args, "backend",  "mdt")
    prefix   = getattr(args, "prefix",   "")
    max_keys = getattr(args, "max",      20)
    print(f"Listing S3 objects on JupyterHub (backend={backend}, prefix='{prefix}')...\n")
    code = _S3_LS_CODE.format(backend=backend, prefix=prefix, max_keys=max_keys)
    _run(code)


# ---------------------------------------------------------------------------
# Command: s3-stream-test  — verify rasterio streaming works from JupyterHub
# ---------------------------------------------------------------------------

_S3_STREAM_CODE = """
import boto3, os, warnings, numpy as np
import rasterio
from rasterio.session import AWSSession
warnings.filterwarnings('ignore')

ep2  = os.getenv("AWS_ENDPOINT_URL2",    "https://stor-002.a.acnca.pt:9000")
key2 = os.getenv("AWS_ACCESS_KEY_ID2")
sec2 = os.getenv("AWS_SECRET_ACCESS_KEY2")

# Find a real MDT-50cm tile via STAC
import urllib.request, json as _json, ssl as _ssl
ctx = _ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=_ssl.CERT_NONE
req = urllib.request.urlopen("https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items?limit=1",
                             context=ctx)
data = _json.loads(req.read())
href = data["features"][0]["assets"]["Data"]["href"]
# href is https://stor-002.a.acnca.pt:9000/lidar/MDT50cm/xxx.tif
key_path = "/".join(href.split("/")[4:])   # strip https://host:port/bucket/
vsis3_path = f"/vsis3/{os.getenv('AWS_S3_BUCKET','lidar')}/{key_path}"

print(f"STAC item href : {href}")
print(f"GDAL vsis3 path: {vsis3_path}")
print()

sess = boto3.Session(aws_access_key_id=key2, aws_secret_access_key=sec2)
env = rasterio.Env(
    AWSSession(sess),
    AWS_S3_ENDPOINT   = ep2.replace("https://","").replace("http://","").rstrip("/"),
    AWS_HTTPS         = "YES",
    AWS_VIRTUAL_HOSTING = "FALSE",
    GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR",
)
with env:
    with rasterio.open(vsis3_path) as src:
        print(f"CRS     : {src.crs}")
        print(f"Size    : {src.width} x {src.height} px  ({src.res[0]:.2f} m resolution)")
        print(f"Bounds  : {src.bounds}")
        print(f"Nodata  : {src.nodata}")
        print(f"Dtype   : {src.dtypes[0]}")
        arr = src.read(1, out_shape=(200, 200))
        valid = arr[arr != src.nodata] if src.nodata is not None else arr.ravel()
        if len(valid):
            print(f"Elev range (preview): {valid.min():.2f} to {valid.max():.2f} m")
        print()
        print("STREAMING OK — rasterio can read MDT-50cm directly from MinIO!")
"""

def cmd_s3_stream_test(_args) -> None:
    print("Testing rasterio streaming of MDT-50cm from MinIO on JupyterHub...\n")
    _run(_S3_STREAM_CODE)


# ---------------------------------------------------------------------------
# Command: sync-project  — push local project files to JupyterHub
# ---------------------------------------------------------------------------

def cmd_sync_project(args) -> None:
    remote_base = getattr(args, "remote_path", None) or "reef-imagery-pipeline"
    extensions  = tuple((getattr(args, "ext", None) or "py,ipynb,toml,md,txt").split(","))
    extensions  = tuple(f".{e.lstrip('.')}" for e in extensions)
    dry_run     = getattr(args, "dry_run", False)

    # Gather local files to sync
    ignore_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules",
                   ".ipynb_checkpoints", "sentinel_images", "reef_Output_Master"}
    files_to_sync: list[tuple[Path, str]] = []

    for f in PROJECT_ROOT.rglob("*"):
        if f.is_dir():
            continue
        if any(p in ignore_dirs for p in f.parts):
            continue
        if f.suffix not in extensions:
            continue
        rel = f.relative_to(PROJECT_ROOT)
        remote = f"{remote_base}/{rel.as_posix()}"
        files_to_sync.append((f, remote))

    print(f"Files to sync: {len(files_to_sync)} (ext: {extensions})")
    if dry_run:
        for local, remote in files_to_sync:
            print(f"  {local.relative_to(PROJECT_ROOT)} → {remote}")
        return

    ok = err = 0
    for local, remote in files_to_sync:
        try:
            content_raw = local.read_text(encoding="utf-8", errors="replace")
            if local.suffix == ".ipynb":
                payload = {"type": "notebook", "format": "json",
                           "content": json.loads(content_raw)}
            else:
                payload = {"type": "file", "format": "text", "content": content_raw}
            _put(f"{USER_BASE}/api/contents/{remote}", payload)
            print(f"  ✓  {remote}")
            ok += 1
        except Exception as e:
            print(f"  ✗  {remote}  ({e})", file=sys.stderr)
            err += 1

    print(f"\nSync complete: {ok} uploaded, {err} errors.")


# ---------------------------------------------------------------------------
# Command: clone-repo
# ---------------------------------------------------------------------------

def cmd_clone_repo(args) -> None:
    repo_url = args.repo_url
    dest     = getattr(args, "dest", None) or ""
    clone_cmd = f"git clone {repo_url}" + (f" {dest}" if dest else "")
    code = (
        "import subprocess, sys\n"
        f"r = subprocess.run({clone_cmd!r}.split(), capture_output=True, text=True)\n"
        "print(r.stdout)\n"
        "if r.stderr: print(r.stderr, file=sys.stderr)\n"
        "print('Return code:', r.returncode)"
    )
    print(f"Cloning {repo_url}...\n")
    _run(code)


# ---------------------------------------------------------------------------
# Command: kernels  — list running kernels
# ---------------------------------------------------------------------------

def cmd_kernels(_args) -> None:
    kernels = _get(f"{USER_BASE}/api/kernels").json()
    if not kernels:
        print("No kernels running.")
        return
    print(f"Running kernels: {len(kernels)}")
    for k in kernels:
        print(f"  {k['id']}  name={k['name']}  state={k.get('execution_state','?')}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jupyterhub_control.py",
        description="CLI for the DGT/ACNCA JupyterHub — runs code remotely with S3 access.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # whoami
    sub.add_parser("whoami", help="Show authenticated user info.")

    # kernels
    sub.add_parser("kernels", help="List running kernels on the server.")

    # ls
    ls_p = sub.add_parser("ls", help="List files in the remote Jupyter home.")
    ls_p.add_argument("--path", default="", help="Remote sub-path (default: home root).")

    # upload-notebook
    up_p = sub.add_parser("upload-notebook", help="Upload a local .ipynb to the server.")
    up_p.add_argument("notebook", help="Local path to the .ipynb file.")
    up_p.add_argument("--remote-path", dest="remote_path", default=None,
                      help="Remote path (relative to home). Defaults to filename.")

    # run-code
    rc_p = sub.add_parser("run-code", help="Execute a Python code string on a remote kernel.")
    rc_p.add_argument("code", help="Python code to execute (quote it in the shell).")

    # run-script
    rs_p = sub.add_parser("run-script", help="Run a local Python script on a remote kernel.")
    rs_p.add_argument("script", help="Local path to the .py script to execute.")

    # s3-ls
    sl_p = sub.add_parser("s3-ls", help="List objects in the DGT MinIO/S3 buckets.")
    sl_p.add_argument("--backend", default="mdt", choices=["mdt", "laz", "1", "2"],
                      help="Which S3 backend: 'laz'/'1'=stor-001, 'mdt'/'2'=stor-002 (default: mdt).")
    sl_p.add_argument("--prefix", default="", help="S3 key prefix to filter (e.g. 'MDT2m/').")
    sl_p.add_argument("--max", type=int, default=20, help="Max objects to list (default: 20).")

    # s3-stream-test
    sub.add_parser("s3-stream-test",
                   help="Verify rasterio can stream MDT-50cm from MinIO on the JupyterHub.")

    # sync-project
    sy_p = sub.add_parser("sync-project", help="Push local project files to JupyterHub.")
    sy_p.add_argument("--remote-path", dest="remote_path", default="reef-imagery-pipeline",
                      help="Remote base folder name (default: reef-imagery-pipeline).")
    sy_p.add_argument("--ext", default="py,ipynb,toml,md,txt",
                      help="Comma-separated file extensions to sync (default: py,ipynb,toml,md,txt).")
    sy_p.add_argument("--dry-run", action="store_true",
                      help="Print files that would be synced without uploading.")

    # clone-repo
    cr_p = sub.add_parser("clone-repo", help="Clone a git repository into the remote environment.")
    cr_p.add_argument("repo_url", help="HTTPS URL of the repository.")
    cr_p.add_argument("--dest", default=None, help="Target folder name on the server.")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    dispatch = {
        "whoami":          cmd_whoami,
        "kernels":         cmd_kernels,
        "ls":              cmd_ls,
        "upload-notebook": cmd_upload_notebook,
        "run-code":        cmd_run_code,
        "run-script":      cmd_run_script,
        "s3-ls":           cmd_s3_ls,
        "s3-stream-test":  cmd_s3_stream_test,
        "sync-project":    cmd_sync_project,
        "clone-repo":      cmd_clone_repo,
    }
    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
