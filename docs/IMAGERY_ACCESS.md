# Imagery Access — What Works, What Doesn't

> Verified live on 2026-06-24 against the credentials in `.env`.
> This documents the *actual* access situation, not aspirational plans.

## TL;DR

Your **Sentinel Hub** account is the best usable access you have, and it's already
wired into the pipeline via [`src/sh_downloader.py`](../src/sh_downloader.py). It serves
**Sentinel-2 L2A** and **Sentinel-1 GRD** — which is everything the core SDB/BVI
pipeline needs. You do **not** need the Planet Orders/download API for the core
pipeline, because the pipeline runs on Sentinel-2, not PlanetScope.

The `SH_CLIENT_ID` in `.env` (`0c1a63b2-…`) is the OAuth client originally created as
`planet-algarve-reef`. Its JWT shows `aud: api.planet.com` — Planet owns Sentinel Hub,
so this one client authenticates against Sentinel Hub's Process/Catalog/Statistical APIs.

## Verified status (2026-06-24)

| Capability | Status | How verified |
|---|---|---|
| SH OAuth token (`services.sentinel-hub.com`) | ✅ Works | token endpoint → HTTP 200 |
| SH Process API → Sentinel-2 L2A | ✅ Works | pulled a 32×32 B02/B03/B04 patch, 0.02 PU |
| SH Catalog → Sentinel-2 L2A | ✅ Scenes returned | `catalog/search` HTTP 200 |
| SH Catalog → Sentinel-1 GRD | ✅ Scenes returned | `catalog/search` HTTP 200 |
| SH → PlanetScope / SkySat | ❌ Not subscribed | catalog HTTP 400 |
| Direct Planet API key (Orders/download) | ❌ No download plan | account has no Downloads/Streaming plan |
| `.env` secrets in git | ✅ Safe | gitignored (`.gitignore:29`), not tracked |

### The one caveat: finite quota

`GET /api/v1/accounting/usage` reports `processingUnitsMonthly: configuration 0,
remaining 0` — i.e. **no paid monthly plan**. However, there is a one-time
`processingUnitsOverage` allowance of **30,000 PU** (only 19 consumed as of this
write), which is what the trial/exploration account actually runs on. It works
today but is finite — when the overage runs dry, requests will start failing.
For a sustainable, free, *unlimited-enough* path, see **CDSE** below.

## How to use the working SH path today

```bash
# 1. Load credentials into the environment (sh_downloader reads SH_CLIENT_ID/SECRET
#    and CDSE_CLIENT_ID/SECRET if present).
set -a; source .env; set +a

# 2. Pull the clearest S2-L2A 4-band patch over the Pedra reef.
#    endpoint="auto" (default) tries the commercial SH endpoint first, then
#    automatically falls back to CDSE on any HTTP error.
python - <<'PY'
from src.sh_downloader import download_patch
download_patch(
    lon=-8.2102, lat=37.0691,
    size_m=1280,
    date_range=("2025-09-01", "2025-09-30"),
    bands=["B02", "B03", "B04", "B08"],
    output_path="outputs/reef_patches/pedra_2025-09.tif",
)

# Skip commercial entirely and force the free CDSE endpoint:
download_patch(..., endpoint="cdse")
PY
```

`sh_downloader` also exposes `search_scenes()`, `download_scene(bbox, …)`, and
`get_stats()` (monthly band statistics via the Statistical API). All accept the
same `endpoint="auto" | "commercial" | "cdse"` argument.

## Automatic commercial → CDSE fallback (already wired)

[`src/sh_downloader.py`](../src/sh_downloader.py) supports **two SH endpoints**
behind a single function call:

- **commercial** — `services.sentinel-hub.com` (your existing `SH_CLIENT_*`)
- **cdse**       — `sh.dataspace.copernicus.eu` (`CDSE_CLIENT_*`, free tier)

Each public function (`download_scene`, `download_patch`, `search_scenes`,
`get_stats`) takes `endpoint="auto"` by default. With both credential pairs in
the environment, it tries commercial first and falls back to CDSE on any HTTP
error (401/403/429 quota, 5xx server, or even 400 — the next endpoint is a
different realm entirely, so we don't trust a single error to be fatal).
Per-endpoint token caches mean the two realms coexist cleanly in the same
process.

With only one credential pair set, `endpoint="auto"` simply uses the available
endpoint directly (no fallback). Pass `endpoint="commercial"` or `endpoint="cdse"`
explicitly to bypass the resolver.

## Best-access strategy, by what each sensor uniquely offers

### Sentinel-2 / Sentinel-1 (10 m — the pipeline core)
Already free and working via SH, with automatic CDSE fallback when the trial
quota runs dry. Additional free paths in the repo:
- `src/stac_ingest.py` — Earth Search (AWS) + Planetary Computer, both free, no quota.
- `src/sh_downloader.py` — commercial SH now with **built-in CDSE fallback**.

### Sustainable free upgrade — CDSE Sentinel Hub (recommended when trial runs dry)
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) provides Sentinel Hub
**free** (~30,000 PU/month) to anyone with a Copernicus account. Same Process API,
different token + base URL:
- Token: `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`
- Base:  `https://sh.dataspace.copernicus.eu`

Create a free OAuth client at <https://shapps.dataspace.copernicus.eu/dashboard/>
(User settings → OAuth clients). The commercial SH creds in `.env` do **not** work
against CDSE (different realm — tested, HTTP 401); you need a separate CDSE client,
stored as `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` in `.env`. Both pairs can coexist;
`endpoint="auto"` picks commercial first and falls back to CDSE automatically.

### PlanetScope (3 m — the only thing Planet uniquely adds)
Not reachable today by *any* of your credentials (direct Planet key has no download
plan; SH account isn't subscribed to PlanetScope). For a Nova IMS / Portugal reef-
research project, the realistic **free** routes are applications, not API toggles:
- **ESA Network of Resources (NoR) sponsorship** — <https://nor-discover.org/> — free
  commercial data (incl. Planet) for European research projects. Best fit for this work.
- **Planet Education & Research** — <https://www.planet.com/markets/education-and-research/>
  — free non-commercial quota (~5,000 km²/month).
- Buying a Planet download plan is the only *instant* option, and likely unnecessary
  given the pipeline is Sentinel-2-based.

A search-only Planet script already exists at
[`scripts/fetch_planet_highres.py`](../scripts/fetch_planet_highres.py); it needs
`PLANET_API_KEY` in the environment and will return metadata even without a download
plan, but the per-band HTTP reads in it require download entitlement to succeed.
