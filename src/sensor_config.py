#!/usr/bin/env python3
"""
sensor_config.py — Multi-Sensor Configuration Registry
══════════════════════════════════════════════════════════════════════════════════
Centralized sensor definitions for satellite imagery processing.
Maps sensor name → bands, wavelengths, resolutions, DN scaling, ACOLITE config.

Supported sensors:
  - sentinel-2  (ESA, 10m, free)
  - pleiades-neo (Airbus, 0.3m/1.2m, commercial/ESA TPM)
  - spot-6/7    (Airbus, 1.5m/6m, commercial/ESA TPM)
  - worldview-3  (Maxar, 0.31m/1.24m, commercial)

Usage:
  from src.sensor_config import get_sensor, list_sensors, SENSOR_REGISTRY
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BandConfig:
    """Single band configuration."""
    name: str              # Asset key in STAC or local file
    wavelength_nm: int     # Center wavelength in nm
    resolution_m: float    # Ground sampling distance in meters
    description: str       # Human-readable name
    reef_priority: int     # 1=critical, 2=useful, 3=optional for reef ID
    aw: float = 0.0        # Pure-water absorption coefficient (Pope & Fry 1997)


@dataclass
class SensorConfig:
    """Complete sensor configuration."""
    name: str                        # e.g. "sentinel-2"
    display_name: str                # e.g. "Sentinel-2 MSI"
    stac_collection: str             # STAC collection ID
    provider: str                    # "microsoft_pc", "cdse", "airbus", "maxar"
    bands: Dict[str, BandConfig]     # band_key → BandConfig
    dn_scale: float                  # DN-to-reflectance divisor (e.g. 10000 for S2)
    acolite_sensor: str              # ACOLITE --sensor flag value
    crs: str                         # Default CRS for processing
    min_wavelength_nm: int           # Shortest useful wavelength
    max_wavelength_nm: int           # Longest useful wavelength
    has_swir: bool                   # Has SWIR bands (useful for cloud/water masking)
    has_red_edge: bool               # Has red-edge bands (useful for submerged vegetation)
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# BAND DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Pure water absorption coefficients (m⁻¹) — Pope & Fry 1997
# These are critical for Beer-Lambert depth estimation
AW_443 = 0.0145   # Coastal aerosol / Blue
AW_490 = 0.0190   # Blue
AW_510 = 0.0590   # Green (short)
AW_555 = 0.0612   # Green
AW_560 = 0.0612   # Green (S2)
AW_640 = 0.2890   # Red (short)
AW_665 = 0.4300   # Red
AW_705 = 0.6500   # Red Edge 1
AW_740 = 1.7700   # Red Edge 2
AW_783 = 2.3100   # Red Edge 3
AW_842 = 3.0600   # NIR
AW_865 = 3.5800   # Narrow NIR
AW_1610 = 85.0    # SWIR1
AW_2190 = 440.0   # SWIR2


# ═══════════════════════════════════════════════════════════════════════════════
# SENSOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

SENTINEL_2 = SensorConfig(
    name="sentinel-2",
    display_name="Sentinel-2 MSI",
    stac_collection="sentinel-2-l2a",
    provider="microsoft_pc",
    dn_scale=10000.0,
    acolite_sensor="S2",
    crs="EPSG:32629",
    min_wavelength_nm=443,
    max_wavelength_nm=2190,
    has_swir=True,
    has_red_edge=True,
    bands={
        "B01": BandConfig("B01", 443, 60,  "Coastal Aerosol", 3, AW_443),
        "B02": BandConfig("B02", 490, 10,  "Blue",            1, AW_490),
        "B03": BandConfig("B03", 560, 10,  "Green",           1, AW_560),
        "B04": BandConfig("B04", 665, 10,  "Red",             2, AW_665),
        "B05": BandConfig("B05", 705, 20,  "Red Edge 1",      3, AW_705),
        "B06": BandConfig("B06", 740, 20,  "Red Edge 2",      3, AW_740),
        "B07": BandConfig("B07", 783, 20,  "Red Edge 3",      3, AW_783),
        "B08": BandConfig("B08", 842, 10,  "NIR",             2, AW_842),
        "B8A": BandConfig("B8A", 865, 20,  "Narrow NIR",      3, AW_865),
        "B11": BandConfig("B11", 1610, 20, "SWIR1",           3, AW_1610),
        "B12": BandConfig("B12", 2190, 20, "SWIR2",           3, AW_2190),
    },
    notes="Baseline sensor. 10m blue/green/red/NIR. Free via Copernicus/Planetary Computer.",
)


PLEIADES_NEO = SensorConfig(
    name="pleiades-neo",
    display_name="Pléiades Neo (HR)",
    stac_collection="pleiades-neo",
    provider="airbus",
    dn_scale=10000.0,   # L2A reflectance product (same scaling as S2)
    acolite_sensor="PLEIADES",
    crs="EPSG:32629",
    min_wavelength_nm=400,
    max_wavelength_nm=1000,
    has_swir=False,
    has_red_edge=False,
    bands={
        "PAN": BandConfig("PAN",  520, 0.3, "Panchromatic",  2, 0.0),
        "B02": BandConfig("B02",  490, 1.2, "Blue",          1, AW_490),   # Aqua band
        "B03": BandConfig("B03",  560, 1.2, "Green",         1, AW_560),
        "B04": BandConfig("B04",  665, 1.2, "Red",           2, AW_665),
        "B08": BandConfig("B08",  842, 1.2, "NIR",           2, AW_842),
    },
    notes=(
        "30cm PAN / 1.2m MS. Best for shallow reef mapping (<10m). "
        "No SWIR or red-edge. Available via Airbus OneAtlas or ESA TPM. "
        "For deeper reef ID, B02 at 1.2m gives 8x better spatial detail than S2."
    ),
)

SPOT_6_7 = SensorConfig(
    name="spot-6/7",
    display_name="SPOT 6/7 (HRG)",
    stac_collection="spot-6/7",
    provider="airbus",
    dn_scale=10000.0,
    acolite_sensor="SPOT",
    crs="EPSG:32629",
    min_wavelength_nm=450,
    max_wavelength_nm=1750,
    has_swir=True,
    has_red_edge=False,
    bands={
        "PAN": BandConfig("PAN",  620, 1.5, "Panchromatic",  2, 0.0),
        "B02": BandConfig("B02",  490, 6.0, "Blue",          1, AW_490),
        "B03": BandConfig("B03",  560, 6.0, "Green",         1, AW_560),
        "B04": BandConfig("B04",  665, 6.0, "Red",           2, AW_665),
        "B08": BandConfig("B08",  842, 6.0, "NIR",           2, AW_842),
        "B11": BandConfig("B11", 1610, 6.0, "SWIR1",         3, AW_1610),
    },
    notes="1.5m PAN / 6m MS. Good balance of resolution and coverage. SWIR helps cloud masking.",
)

WORLDVIEW_3 = SensorConfig(
    name="worldview-3",
    display_name="WorldView-3 (SWIR)",
    stac_collection="worldview-3",
    provider="maxar",
    dn_scale=10000.0,
    acolite_sensor="WV3",
    crs="EPSG:32629",
    min_wavelength_nm=400,
    max_wavelength_nm=2500,
    has_swir=True,
    has_red_edge=False,
    bands={
        "PAN":  BandConfig("PAN",   620, 0.31, "Panchromatic",  2, 0.0),
        "B02":  BandConfig("B02",   480, 1.24, "Coastal Blue",  1, AW_443),
        "B03":  BandConfig("B03",   545, 1.24, "Green",         1, AW_555),
        "B04":  BandConfig("B04",   660, 1.24, "Red",           2, AW_665),
        "B08":  BandConfig("B08",   832, 1.24, "NIR",           2, AW_842),
        "B11":  BandConfig("B11",  1210, 3.70, "SWIR1",         3, AW_1610),
        "B12":  BandConfig("B12",  1570, 3.70, "SWIR2",         3, AW_2190),
        "B_S1": BandConfig("B_S1", 2245, 3.70, "SWIR-7",        3, AW_2190),
    },
    notes=(
        "0.31m PAN / 1.24m MS / 3.7m SWIR. Best commercial sensor for water. "
        "SWIR bands enable sunglint correction and shallow water detection."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# SENSOR REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

SENSOR_REGISTRY: Dict[str, SensorConfig] = {
    "sentinel-2":   SENTINEL_2,
    "pleiades-neo": PLEIADES_NEO,
    "spot-6/7":     SPOT_6_7,
    "worldview-3":  WORLDVIEW_3,
}


def get_sensor(name: str) -> SensorConfig:
    """Get sensor config by name. Raises ValueError if not found."""
    key = name.lower().strip()
    if key not in SENSOR_REGISTRY:
        available = ", ".join(SENSOR_REGISTRY.keys())
        raise ValueError(f"Unknown sensor '{name}'. Available: {available}")
    return SENSOR_REGISTRY[key]


def list_sensors() -> List[str]:
    """Return list of available sensor names."""
    return list(SENSOR_REGISTRY.keys())


def get_reef_bands(sensor_name: str, max_priority: int = 2) -> Dict[str, BandConfig]:
    """Get bands useful for reef identification (priority <= max_priority)."""
    sensor = get_sensor(sensor_name)
    return {k: v for k, v in sensor.bands.items() if v.reef_priority <= max_priority}


def get_blue_band(sensor_name: str) -> BandConfig:
    """Get the primary blue (B02) band for a sensor — the only useful band for reef ID."""
    sensor = get_sensor(sensor_name)
    return sensor.bands["B02"]


def map_band_name(from_sensor: str, to_sensor: str, band_name: str) -> Optional[str]:
    """
    Map a band name from one sensor to another by wavelength proximity.
    Returns None if no match within 50nm.
    """
    src = get_sensor(from_sensor)
    tgt = get_sensor(to_sensor)
    if band_name not in src.bands:
        return None
    src_wl = src.bands[band_name].wavelength_nm
    best_match = None
    best_diff = 999
    for tgt_name, tgt_band in tgt.bands.items():
        diff = abs(tgt_band.wavelength_nm - src_wl)
        if diff < best_diff and diff < 50:
            best_diff = diff
            best_match = tgt_name
    return best_match


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════

def print_sensor_comparison():
    """Print a comparison table of all sensors."""
    print(f"\n{'Sensor':<16} {'Resolution':>10} {'Bands':>6} {'Blue(B02)':>10} {'SWIR':>5} {'RedEdge':>7} {'Cost':>8}")
    print("-" * 70)
    for name, cfg in SENSOR_REGISTRY.items():
        blue_res = cfg.bands["B02"].resolution_m
        n_bands = len(cfg.bands)
        swir = "Yes" if cfg.has_swir else "No"
        red_edge = "Yes" if cfg.has_red_edge else "No"
        cost = "Free" if cfg.provider in ("microsoft_pc", "cdse") else "Commercial"
        print(f"  {cfg.display_name:<14} {blue_res:>7.1f}m {n_bands:>5} {blue_res:>7.1f}m {swir:>5} {red_edge:>7} {cost:>8}")


if __name__ == "__main__":
    print_sensor_comparison()
