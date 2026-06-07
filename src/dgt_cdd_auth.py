"""
dgt_cdd_auth.py — Shared authentication helper for the DGT Centro de Dados.

Both dgt_ortho_client and coastal_topography use the same PKCE + BFF session
flow to authenticate against cdd.dgterritorio.gov.pt.

Usage:
    from src.dgt_cdd_auth import get_cdd_session, get_signed_url

    session = get_cdd_session()           # authenticated requests.Session
    url     = get_signed_url("MDT-50cm", "MDT-50cm-196015-04-2024")

Credentials:  DGT_CDD_USERNAME / DGT_CDD_PASSWORD  (env vars)
Registration: https://cdd.dgterritorio.gov.pt  (free, CC-BY 4.0)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import secrets
from typing import Optional

import requests

log = logging.getLogger(__name__)

_KC_AUTH_URL  = ("https://auth.cdd.dgterritorio.gov.pt/realms/dgterritorio"
                 "/protocol/openid-connect/auth")
_CDD_CLIENT   = "aai-oidc-dgt"
_CDD_CALLBACK = "https://cdd.dgterritorio.gov.pt/auth/callback"
_CDD_BACKEND  = "https://cdd.dgterritorio.gov.pt/dgt-be"

_session: Optional[requests.Session] = None


def _login() -> Optional[requests.Session]:
    """Perform PKCE + BFF flow. Returns an authenticated Session or None."""
    user = os.environ.get("DGT_CDD_USERNAME")
    pwd  = os.environ.get("DGT_CDD_PASSWORD")
    if not user or not pwd:
        log.warning(
            "DGT_CDD_USERNAME / DGT_CDD_PASSWORD not set. "
            "Register free at https://cdd.dgterritorio.gov.pt to download DGT data."
        )
        return None

    sess = requests.Session()
    try:
        # PKCE
        verifier  = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()

        # Step 1 — Keycloak login page
        r = sess.get(_KC_AUTH_URL, params={
            "client_id": _CDD_CLIENT, "response_type": "code",
            "redirect_uri": _CDD_CALLBACK, "scope": "openid profile email",
            "code_challenge": challenge, "code_challenge_method": "S256",
        }, timeout=15, allow_redirects=True)
        r.raise_for_status()

        action = re.search(r'action=["\']([^"\']+)["\']', r.text)
        if not action:
            log.error("DGT CDD: Keycloak login form not found")
            return None

        # Step 2 — submit credentials
        r2 = sess.post(
            action.group(1).replace("&amp;", "&"),
            data={"username": user, "password": pwd, "credentialId": ""},
            timeout=15, allow_redirects=False,
        )
        if r2.status_code not in (301, 302):
            msg = re.search(
                r'(?:alert-error|kc-feedback-text)[^>]*>([^<]+)', r2.text
            )
            log.error("DGT CDD login failed: %s",
                      msg.group(1).strip() if msg else f"HTTP {r2.status_code}")
            return None

        # Step 3 — BFF /auth/callback sets session cookies
        sess.get(r2.headers["Location"], timeout=15, allow_redirects=True)

        if "connect.sid" not in sess.cookies and "auth_session" not in sess.cookies:
            log.error("DGT CDD: BFF did not return session cookies")
            return None

        log.info("DGT CDD: authenticated (session cookies obtained)")
        return sess

    except Exception as exc:
        log.warning("DGT CDD authentication error: %s", exc)
        return None


def get_cdd_session(force_refresh: bool = False) -> Optional[requests.Session]:
    """Return a cached authenticated Session, logging in if needed."""
    global _session
    if force_refresh or _session is None or "connect.sid" not in _session.cookies:
        _session = _login()
    return _session


def invalidate() -> None:
    """Force re-login on the next get_cdd_session() call."""
    global _session
    _session = None


def get_signed_url(collection: str, item_id: str) -> Optional[str]:
    """
    Return a signed download URL for a DGT STAC item.

    Calls the authenticated CDD backend which returns an asset href of the form:
        /dgt-be/v1/download/{sha256}
    that resolves directly to the COG file.

    Returns absolute URL string, or None on failure.
    """
    sess = get_cdd_session()
    if sess is None:
        return None

    url = f"{_CDD_BACKEND}/v1/collections/{collection}/items/{item_id}"
    try:
        r = sess.get(url, timeout=15)
        if r.status_code == 401:
            invalidate()
            sess = get_cdd_session()
            if sess is None:
                return None
            r = sess.get(url, timeout=15)

        if r.status_code != 200:
            log.warning("CDD item lookup failed: HTTP %d — %s/%s",
                        r.status_code, collection, item_id)
            return None

        assets = r.json().get("data", {}).get("assets", {})
        asset  = assets.get("visual") or assets.get("Data") or assets.get("data")
        if not asset:
            return None

        href = asset.get("href", "")
        if href.startswith("/dgt-be"):
            href = "https://cdd.dgterritorio.gov.pt" + href
        return href

    except Exception as exc:
        log.warning("CDD signed URL error for %s/%s: %s", collection, item_id, exc)
        return None
