"""
cmems_kd490.py — Live Kd490 and Secchi depth climatology from CMEMS.

Dataset: cmems_obs-oc_atl_bgc-transp_my_l3-multi-1km_P1D
  Atlantic, 1 km, daily, multi-sensor reprocessed (MY).
  Variables used: KD490, ZSD (Secchi disk depth).

Authentication (in priority order):
  1. ~/.copernicusmarine/.copernicusmarine-credentials  (set via `copernicusmarine login`)
  2. CMEMS_USER + CMEMS_PASSWORD environment variables
  3. COPERNICUSMARINE_SERVICE_USERNAME + _PASSWORD environment variables

Falls back silently to the static KD490_TABLE from constants.py when
copernicusmarine is unavailable or credentials cannot be resolved.

Usage:
    from src.cmems_kd490 import get_kd490, get_zsd, KD490_TABLE_LIVE, ZSD_TABLE_LIVE

    kd  = get_kd490(7)   # July Kd490 (m⁻¹) for Algarve
    zsd = get_zsd(7)     # July Secchi depth (m) — None if CMEMS not loaded
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

log = logging.getLogger(__name__)

# Algarve coastal bounding box
_LON_MIN, _LON_MAX = -9.5, -7.5
_LAT_MIN, _LAT_MAX = 36.8, 37.5

# Atlantic 1 km daily L3 — replaces retired atl_bgc-transp_my_l4-multi-4km_P1M
_DATASET_ID = "cmems_obs-oc_atl_bgc-transp_my_l3-multi-1km_P1D"

from src.constants import KD490_TABLE as _STATIC_TABLE, KD490_DEFAULT


def _fetch_cmems_climatology() -> tuple[Dict[int, float], Dict[int, float]]:
    """
    Download daily Kd490 + ZSD from CMEMS, aggregate to monthly climatology.

    Returns (kd490_table, zsd_table) where each is {month: value}.
    zsd_table values are Secchi depth in metres (higher = clearer water).
    """
    import copernicusmarine
    import numpy as np

    user = os.environ.get("CMEMS_USER") or os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    pwd  = os.environ.get("CMEMS_PASSWORD") or os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

    log.info("Fetching CMEMS Kd490+ZSD climatology (dataset=%s) …", _DATASET_ID)

    kwargs: dict = dict(
        dataset_id=_DATASET_ID,
        variables=["KD490", "ZSD"],
        minimum_longitude=_LON_MIN,
        maximum_longitude=_LON_MAX,
        minimum_latitude=_LAT_MIN,
        maximum_latitude=_LAT_MAX,
    )
    if user and pwd:
        kwargs["username"] = user
        kwargs["password"] = pwd

    ds = copernicusmarine.open_dataset(**kwargs)

    def _monthly_median(da) -> Dict[int, float]:
        table: Dict[int, float] = {}
        if "time" not in da.dims:
            vals = da.values.ravel()
            valid = vals[np.isfinite(vals) & (vals > 0)]
            v = float(np.nanmedian(valid)) if valid.size > 0 else 0.0
            return {m: v for m in range(1, 13)}
        monthly = da.groupby("time.month").median(dim="time")
        for m in range(1, 13):
            if m not in monthly.month.values:
                continue
            vals = monthly.sel(month=m).values.ravel()
            valid = vals[np.isfinite(vals) & (vals > 0)]
            if valid.size > 0:
                table[m] = float(np.nanmedian(valid))
        return table

    kd_table  = _monthly_median(ds["KD490"])
    zsd_table = _monthly_median(ds["ZSD"])

    # Fill missing KD490 months from static table
    for m in range(1, 13):
        if m not in kd_table:
            kd_table[m] = _STATIC_TABLE.get(m, KD490_DEFAULT)
            log.debug("Month %d KD490 missing — static fallback %.4f", m, kd_table[m])

    log.info(
        "CMEMS loaded — KD490: Jul=%.4f Aug=%.4f Sep=%.4f (static Jul=%.4f) | "
        "ZSD: Jul=%.1fm Aug=%.1fm Sep=%.1fm",
        kd_table.get(7, 0), kd_table.get(8, 0), kd_table.get(9, 0),
        _STATIC_TABLE.get(7, 0),
        zsd_table.get(7, 0), zsd_table.get(8, 0), zsd_table.get(9, 0),
    )
    return kd_table, zsd_table


# Module-level tables — static at import; updated by refresh_from_cmems()
KD490_TABLE_LIVE: Dict[int, float] = dict(_STATIC_TABLE)
ZSD_TABLE_LIVE:   Dict[int, float] = {}   # empty until first successful refresh


def refresh_from_cmems() -> bool:
    """
    Fetch live Kd490 + ZSD from CMEMS and update module-level tables.

    Uses ~/.copernicusmarine/ stored credentials automatically; env vars override.
    Returns True on success, False on any failure (tables retain previous values).
    """
    global KD490_TABLE_LIVE, ZSD_TABLE_LIVE
    try:
        kd, zsd = _fetch_cmems_climatology()
        KD490_TABLE_LIVE = kd
        ZSD_TABLE_LIVE   = zsd
        return True
    except ImportError:
        log.warning("copernicusmarine not installed — retaining static Kd490 table.")
    except Exception as e:
        log.warning("CMEMS fetch failed (%s: %s) — retaining static table.",
                    type(e).__name__, e)
    return False


def get_kd490(month: int) -> float:
    """Kd490 (m⁻¹) for the given calendar month. Uses CMEMS if loaded, static otherwise."""
    return KD490_TABLE_LIVE.get(month, KD490_DEFAULT)


def get_zsd(month: int) -> Optional[float]:
    """
    Secchi disk depth (m) for the given calendar month from CMEMS.

    Returns None if CMEMS has not been loaded (refresh_from_cmems() not yet called
    or failed). Higher value = clearer water.
    """
    return ZSD_TABLE_LIVE.get(month)
