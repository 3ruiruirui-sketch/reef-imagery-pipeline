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
from html.parser import HTMLParser
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

_KC_AUTH_HOST = "https://auth.cdd.dgterritorio.gov.pt"
_KC_AUTH_URL = f"{_KC_AUTH_HOST}/realms/dgterritorio" "/protocol/openid-connect/auth"
_CDD_CLIENT = "aai-oidc-dgt"
_CDD_CALLBACK = "https://cdd.dgterritorio.gov.pt/auth/callback"
_CDD_BACKEND = "https://cdd.dgterritorio.gov.pt/dgt-be"
_CDD_MAIN = "https://cdd.dgterritorio.gov.pt"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
    "Upgrade-Insecure-Requests": "1",
}

_session: requests.Session | None = None


class _KeycloakFormParser(HTMLParser):
    """Extract the action URL and hidden inputs from Keycloak's <form id="kc-form-login">."""

    def __init__(self) -> None:
        super().__init__()
        self.form_action: str | None = None
        self.form_data: dict[str, str] = {}
        self._in_form = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form" and a.get("id") == "kc-form-login":
            self._in_form = True
            self.form_action = a.get("action")
        elif tag == "input" and self._in_form:
            name = a.get("name")
            if name and a.get("type", "text") == "hidden":
                self.form_data[name] = a.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form" and self._in_form:
            self._in_form = False


def _login_keycloak_direct() -> requests.Session | None:
    """
    Authenticate via direct Keycloak form POST (no PKCE).

    Mirrors the flow used by the community QGIS plugin `dgt_cdd_downloader`,
    which is known to produce session cookies that work for both `/dgt-be/v1/*`
    endpoints AND direct asset href downloads.

    Steps:
      1. GET main site to seed cookies.
      2. GET Keycloak auth endpoint (without PKCE) to load the login page.
      3. Parse <form id="kc-form-login"> for action URL and ALL hidden inputs.
      4. POST username/password + hidden fields with allow_redirects=True so
         requests follows the entire OIDC → /auth/callback chain.
      5. Success when the final URL lands back on the main site.
    """
    user = os.environ.get("DGT_CDD_USERNAME")
    pwd = os.environ.get("DGT_CDD_PASSWORD")
    if not user or not pwd:
        log.warning(
            "DGT_CDD_USERNAME / DGT_CDD_PASSWORD not set. "
            "Register free at https://cdd.dgterritorio.gov.pt to download DGT data."
        )
        return None

    sess = requests.Session()
    sess.headers.update(_BROWSER_HEADERS)

    try:
        # Step 1 — seed cookies on the main site
        r0 = sess.get(_CDD_MAIN, timeout=30)
        r0.raise_for_status()

        # Step 2 — Keycloak auth endpoint (no PKCE)
        auth_params = {
            "client_id": _CDD_CLIENT,
            "response_type": "code",
            "redirect_uri": _CDD_CALLBACK,
            "scope": "openid profile email",
        }
        r1 = sess.get(f"{_KC_AUTH_URL}?{urlencode(auth_params)}", timeout=30)
        r1.raise_for_status()

        # Step 3 — parse login form (action + hidden inputs)
        parser = _KeycloakFormParser()
        parser.feed(r1.text)
        if not parser.form_action:
            log.error("DGT CDD (direct): kc-form-login not found on auth page")
            return None

        action = parser.form_action.replace("&amp;", "&")
        if action.startswith("/"):
            action = _KC_AUTH_HOST + action

        # Step 4 — submit credentials, follow the full redirect chain
        login_data = dict(parser.form_data)
        login_data.update({"username": user, "password": pwd, "credentialId": ""})

        login_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": _KC_AUTH_HOST,
            "Referer": r1.url,
        }
        r2 = sess.post(action, data=login_data, headers=login_headers, allow_redirects=True, timeout=30)

        # Step 5 — success criterion: ended up back on the main site
        if not r2.url.startswith(_CDD_MAIN):
            msg = re.search(r"(?:alert-error|kc-feedback-text)[^>]*>([^<]+)", r2.text)
            log.error("DGT CDD (direct) login failed: %s", msg.group(1).strip() if msg else f"final URL {r2.url}")
            return None

        if not any(c.name in {"connect.sid", "auth_session", "JSESSIONID"} for c in sess.cookies):
            log.warning("DGT CDD (direct): no recognised session cookie after auth")

        log.info("DGT CDD: authenticated via direct Keycloak flow")
        return sess

    except requests.RequestException as exc:
        log.warning("DGT CDD (direct) network error: %s", exc)
        return None
    except Exception as exc:
        log.warning("DGT CDD (direct) authentication error: %s", exc)
        return None


def _login_pkce() -> requests.Session | None:
    """Perform PKCE + BFF flow. Returns an authenticated Session or None.

    Kept as a fallback. The direct Keycloak flow (see _login_keycloak_direct)
    is preferred because its cookies also work for asset href downloads.
    """
    user = os.environ.get("DGT_CDD_USERNAME")
    pwd = os.environ.get("DGT_CDD_PASSWORD")
    if not user or not pwd:
        log.warning(
            "DGT_CDD_USERNAME / DGT_CDD_PASSWORD not set. "
            "Register free at https://cdd.dgterritorio.gov.pt to download DGT data."
        )
        return None

    sess = requests.Session()
    try:
        # PKCE
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

        # Step 1 — Keycloak login page
        r = sess.get(
            _KC_AUTH_URL,
            params={
                "client_id": _CDD_CLIENT,
                "response_type": "code",
                "redirect_uri": _CDD_CALLBACK,
                "scope": "openid profile email",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
            timeout=15,
            allow_redirects=True,
        )
        r.raise_for_status()

        action = re.search(r'action=["\']([^"\']+)["\']', r.text)
        if not action:
            log.error("DGT CDD: Keycloak login form not found")
            return None

        # Step 2 — submit credentials
        r2 = sess.post(
            action.group(1).replace("&amp;", "&"),
            data={"username": user, "password": pwd, "credentialId": ""},
            timeout=15,
            allow_redirects=False,
        )
        if r2.status_code not in (301, 302):
            msg = re.search(r"(?:alert-error|kc-feedback-text)[^>]*>([^<]+)", r2.text)
            log.error("DGT CDD login failed: %s", msg.group(1).strip() if msg else f"HTTP {r2.status_code}")
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


def _is_session_alive(sess: requests.Session) -> bool:
    return any(c.name in {"connect.sid", "auth_session", "JSESSIONID"} for c in sess.cookies)


def get_cdd_session(force_refresh: bool = False) -> requests.Session | None:
    """Return a cached authenticated Session, logging in if needed.

    Tries the direct Keycloak flow first; falls back to PKCE/BFF if it fails.
    """
    global _session
    if not force_refresh and _session is not None and _is_session_alive(_session):
        return _session

    sess = _login_keycloak_direct()
    if sess is None:
        log.info("DGT CDD: direct flow failed, trying PKCE fallback")
        sess = _login_pkce()
    _session = sess
    return _session


def invalidate() -> None:
    """Force re-login on the next get_cdd_session() call."""
    global _session
    _session = None


def get_signed_url(collection: str, item_id: str) -> str | None:
    """
    Return a download URL for a DGT STAC item via the authenticated CDD backend.

    The CDD proxy embeds a ready-to-use download href directly in the item's
    ``data`` asset, of the form::

        https://cdd.dgterritorio.gov.pt/dgt-be/v1/download/{sha256}

    There is NO separate token-generation step — the sha256 is pre-computed and
    GETting that URL with the session cookie streams the file.  Collections that
    are mere passthroughs of the public STAC (e.g. the orthophotos) instead
    expose a ``visual`` asset pointing at the *raw private MinIO bucket*
    (stor-002.a.acnca.pt:9000/...), which returns HTTP 403 AccessDenied and is
    therefore not directly downloadable.  We prefer the proxied ``data`` asset
    and only fall back to ``visual`` when no proxied href exists.

    Available CDD collections (verified 2026-06): MDT-50cm, MDT-2m, MDS-50cm,
    MDS-2m, LAZ.  Orthophotos are NOT in this catalogue.

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
            log.warning("CDD item lookup failed: HTTP %d — %s/%s", r.status_code, collection, item_id)
            return None

        assets = r.json().get("data", {}).get("assets", {})
        # Prefer the proxied download asset ("data"); the raw-bucket "visual"
        # asset is a last resort (returns 403 for private CDD buckets).
        asset = assets.get("data") or assets.get("Data") or assets.get("visual")
        if not asset:
            return None

        href = asset.get("href", "")
        if not href:
            return None
        if href.startswith("/dgt-be"):
            href = "https://cdd.dgterritorio.gov.pt" + href
        if "stor-002" in href and "/dgt-be/v1/download/" not in href:
            log.warning(
                "CDD %s/%s exposes only a raw private-bucket URL (no proxy " "download) — not downloadable via CDD.",
                collection,
                item_id,
            )
        return href

    except Exception as exc:
        log.warning("CDD signed URL error for %s/%s: %s", collection, item_id, exc)
        return None
