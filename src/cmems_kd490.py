"""
cmems_kd490.py — Live Kd490 climatology from CMEMS via copernicusmarine client.

Replaces the static KD490_TABLE in constants.py with satellite-derived monthly
Kd490 values for the Algarve coastal bounding box.

Credentials: set CMEMS_USER and CMEMS_PASSWORD environment variables.
If credentials are absent or CMEMS is unreachable, the static table from
constants.py is used as fallback and a warning is logged.

Usage:
    from src.cmems_kd490 import get_kd490, KD490_TABLE_LIVE

    kd = get_kd490(7)          # July Kd490 for Algarve
    table = KD490_TABLE_LIVE   # {month: kd490} dict, same shape as constants.KD490_TABLE
"""

from __future__ import annotations

import logging
import os
from typing import Dict

log = logging.getLogger(__name__)

# Algarve coastal bounding box for Kd490 spatial average
_LON_MIN, _LON_MAX = -9.5, -7.5
_LAT_MIN, _LAT_MAX = 36.8, 37.5

# CMEMS dataset — ocean colour, Atlantic, 4 km monthly climatology
_DATASET_ID = "cmems_obs-oc_atl_bgc-transp_my_l4-multi-4km_P1M"
_VARIABLE = "KD490"

# Fallback: static seasonal table from constants.py (hand-calibrated for Algarve)
from src.constants import KD490_TABLE as _STATIC_TABLE, KD490_DEFAULT


def _fetch_cmems_climatology() -> Dict[int, float]:
    """
    Download monthly Kd490 climatology from CMEMS and return {month: kd490}.

    Requires CMEMS_USER and CMEMS_PASSWORD environment variables.
    Raises RuntimeError if credentials missing or fetch fails.
    """
    user = os.environ.get("CMEMS_USER") or os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    pwd  = os.environ.get("CMEMS_PASSWORD") or os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")

    if not user or not pwd:
        raise RuntimeError(
            "CMEMS credentials not set. Export CMEMS_USER and CMEMS_PASSWORD "
            "to use live Kd490 data."
        )

    import copernicusmarine  # installed as a dependency

    log.info("Fetching CMEMS Kd490 climatology (dataset=%s) …", _DATASET_ID)

    ds = copernicusmarine.open_dataset(
        dataset_id=_DATASET_ID,
        variables=[_VARIABLE],
        minimum_longitude=_LON_MIN,
        maximum_longitude=_LON_MAX,
        minimum_latitude=_LAT_MIN,
        maximum_latitude=_LAT_MAX,
        username=user,
        password=pwd,
    )

    import numpy as np

    kd_var = ds[_VARIABLE]  # shape: (time, lat, lon) or (lat, lon)

    # Build monthly climatology: group by calendar month, spatial mean
    table: Dict[int, float] = {}
    if "time" in kd_var.dims:
        monthly = kd_var.groupby("time.month").mean(dim="time")
        available_months = set(int(v) for v in monthly.month.values)
        for m in range(1, 13):
            if m not in available_months:
                continue  # filled from static table in the loop below
            vals = monthly.sel(month=m).values.ravel()
            valid = vals[np.isfinite(vals) & (vals > 0)]
            if valid.size > 0:
                table[m] = float(np.nanmedian(valid))
    else:
        # Static spatial slice — no time dimension
        vals = kd_var.values.ravel()
        valid = vals[np.isfinite(vals) & (vals > 0)]
        fallback_kd = float(np.nanmedian(valid)) if valid.size > 0 else KD490_DEFAULT
        table = {m: fallback_kd for m in range(1, 13)}

    # Fill any missing months from the static table
    for m in range(1, 13):
        if m not in table:
            table[m] = _STATIC_TABLE.get(m, KD490_DEFAULT)
            log.debug("Month %d Kd490 missing from CMEMS — using static fallback %.4f", m, table[m])

    log.info(
        "CMEMS Kd490 climatology loaded: "
        "Jul=%.4f Aug=%.4f Sep=%.4f (static: Jul=%.4f)",
        table.get(7, 0), table.get(8, 0), table.get(9, 0),
        _STATIC_TABLE.get(7, 0),
    )
    return table


def _build_table() -> Dict[int, float]:
    """Try CMEMS; fall back to static table on any failure."""
    try:
        return _fetch_cmems_climatology()
    except RuntimeError as e:
        log.warning("CMEMS Kd490 unavailable (%s) — using static table.", e)
    except ImportError:
        log.warning("copernicusmarine not installed — using static Kd490 table.")
    except Exception as e:
        log.warning("CMEMS Kd490 fetch failed (%s: %s) — using static table.",
                    type(e).__name__, e)
    return dict(_STATIC_TABLE)


# Module-level table — initialised from the static table at import time so that
# importing this module never makes a network connection.  Call
# refresh_from_cmems() explicitly (e.g. in the orchestrator at start-up) to
# populate from CMEMS when credentials are available.
KD490_TABLE_LIVE: Dict[int, float] = dict(_STATIC_TABLE)


def refresh_from_cmems() -> bool:
    """
    Attempt to populate KD490_TABLE_LIVE from CMEMS.

    Returns True on success, False if credentials are missing, copernicusmarine
    is not installed, or the fetch fails.  KD490_TABLE_LIVE is not modified on
    failure — the static fallback values are retained.
    """
    global KD490_TABLE_LIVE
    try:
        result = _fetch_cmems_climatology()
        KD490_TABLE_LIVE = result
        return True
    except RuntimeError as e:
        log.warning("CMEMS Kd490 unavailable (%s) — retaining static table.", e)
    except ImportError:
        log.warning("copernicusmarine not installed — retaining static Kd490 table.")
    except Exception as e:
        log.warning("CMEMS Kd490 fetch failed (%s: %s) — retaining static table.",
                    type(e).__name__, e)
    return False


def get_kd490(month: int) -> float:
    """
    Return Kd490 (m⁻¹) for the given calendar month (1=Jan … 12=Dec).

    Uses KD490_TABLE_LIVE (static table by default; CMEMS values after a
    successful refresh_from_cmems() call).
    """
    m = int(month)
    return KD490_TABLE_LIVE.get(m, KD490_DEFAULT)
