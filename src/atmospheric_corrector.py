#!/usr/bin/env python3
"""
atmospheric_corrector.py — Multi-Sensor Atmospheric Correction
══════════════════════════════════════════════════════════════════════════════════
Provides atmospheric correction for satellite imagery from multiple sensors.

For Sentinel-2 L2A: data is already corrected (BOA reflectance), just DN scaling.
For Pléiades Neo L2A: Airbus provides radiometrically corrected products, DN scaling.
For raw L1 data: simplified 6S-like correction using dark-object subtraction (DOS).

Key methods:
  - DOS1 (Dark Object Subtraction): subtracts atmospheric path radiance
  - ACOLITE-compatible: wraps ACOLITE CLI for supported sensors
  - Empirical sunglint: subtracts NIR-leakage from visible bands

Usage:
  from src.atmospheric_corrector import AtmosphericCorrector
  ac = AtmosphericCorrector("pleiades-neo")
  boa = ac.dos1_correct(dn_array, band_name)
"""
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

from src.sensor_config import get_sensor, SensorConfig


class AtmosphericCorrector:
    """
    Multi-sensor atmospheric correction engine.

    Supports:
      - Sentinel-2 L2A (passthrough — already corrected)
      - Pléiades Neo L2A (passthrough — already corrected)
      - Any L1 data via DOS1 (Dark Object Subtraction)
      - ACOLITE CLI wrapper for supported sensors
    """

    # Typical path radiance (Lpath) values for coastal Algarve at sea level
    # Units: reflectance (dimensionless)
    # Source: 6S radiative transfer for mid-latitude summer, maritime aerosols
    # SZA ~30-45° typical for Algarve summer
    PATH_RADIANCE = {
        443: 0.0250,  # Coastal/Blue (highest atmospheric scattering)
        490: 0.0200,  # Blue
        510: 0.0150,  # Green-short
        545: 0.0100,  # Green (WV3)
        560: 0.0090,  # Green
        620: 0.0060,  # PAN center
        640: 0.0055,  # Red-short
        665: 0.0040,  # Red
        705: 0.0030,  # Red Edge 1
        740: 0.0025,  # Red Edge 2
        783: 0.0020,  # Red Edge 3
        832: 0.0015,  # NIR (WV3)
        842: 0.0015,  # NIR
        865: 0.0012,  # Narrow NIR
        1210: 0.0005, # SWIR (WV3)
        1610: 0.0003, # SWIR1
        2190: 0.0001, # SWIR2
        2245: 0.0001, # SWIR7 (WV3)
    }

    def __init__(self, sensor_name: str):
        self.sensor = get_sensor(sensor_name)
        self.sensor_name = sensor_name

    def is_l2a(self) -> bool:
        """Check if sensor data is typically L2A (already atmospherically corrected)."""
        # Sentinel-2 L2A and Pléiades Neo L2A are already corrected
        return self.sensor.provider in ("microsoft_pc", "cdse") or "l2a" in self.sensor.stac_collection.lower()

    def dn_to_reflectance(self, dn_array: np.ndarray, band_name: str) -> np.ndarray:
        """
        Convert DN to BOA (Bottom-of-Atmosphere) reflectance.

        For L2A products: simple DN scaling.
        For L1 products: DOS1 atmospheric correction.
        """
        if self.is_l2a():
            # L2A: just scale DN
            return np.clip(dn_array / self.sensor.dn_scale, 0, 1.5)
        else:
            # L1: apply DOS1
            return self.dos1_correct(dn_array, band_name)

    def dos1_correct(self, dn_array: np.ndarray, band_name: str) -> np.ndarray:
        """
        Dark Object Subtraction (DOS1) atmospheric correction.

        Theory: L_TOA = L_path + T * L_surface
        Where:
          L_TOA   = top-of-atmosphere radiance (from DN)
          L_path  = atmospheric path radiance (dark object value)
          T       = atmospheric transmittance (~0.85 for coastal)
          L_surface = bottom-of-atmosphere radiance (what we want)

        DOS1 assumes the darkest pixel in the scene has zero surface reflectance,
        so its radiance is entirely atmospheric path radiance.

        For reef imagery: we use pre-computed path radiance for Algarve conditions.
        """
        if band_name not in self.sensor.bands:
            raise ValueError(f"Band {band_name} not found in sensor {self.sensor_name}")

        band_cfg = self.sensor.bands[band_name]
        wl = band_cfg.wavelength_nm

        # Get path radiance for this wavelength
        lpath = self._get_path_radiance(wl)

        # Atmospheric transmittance (approximate for coastal conditions)
        # Beer-Lambert: T = exp(-tau * m) where tau is optical depth, m is air mass
        # For mid-latitude, SZA ~35°, m ~1.22
        # tau decreases with wavelength (more scattering at blue)
        tau = self._optical_depth(wl)
        air_mass = 1.22  # ~35° SZA
        transmittance = np.exp(-tau * air_mass)

        # Convert DN to top-of-atmosphere reflectance
        toa = np.clip(dn_array / self.sensor.dn_scale, 0, 1.5)

        # Subtract path radiance and correct for transmittance
        boa = (toa - lpath) / max(transmittance, 0.1)
        boa = np.clip(boa, 0, 1.0)

        # Apply water-leaving reflectance correction
        # In coastal waters, ~10% of BOA is water-leaving signal (good!)
        # We don't subtract it — it's the reef signal we want

        return boa.astype(np.float32)

    def empirical_sunglint(self, b02: np.ndarray, b_nir: np.ndarray,
                           strength: float = 0.6) -> np.ndarray:
        """
        Empirical sunglint correction using NIR band.

        Theory: NIR (842nm) is absorbed by water, so any NIR signal over water
        is due to sunglint. We subtract a scaled NIR from visible bands.

        Formula: R_corrected = R_vis - k * R_NIR
        Where k is the ratio of glint contribution in visible vs NIR.

        For Pléiades Neo: no SWIR available, so NIR is the best proxy.
        For Sentinel-2: SWIR (B11) would be better, but NIR works OK.
        """
        if b_nir is None:
            return b02

        # Estimate glint fraction from NIR
        # Over deep clear water, NIR should be ~0.005-0.015
        # Anything above that is glint
        nir_water_threshold = 0.015
        glint = np.clip(b_nir - nir_water_threshold, 0, 1.0)

        # Scale by empirical factor (sensor-dependent)
        # Blue band has more Rayleigh scattering, so glint contribution is lower
        k = strength * (self.sensor.bands["B02"].wavelength_nm / 842.0)  # ~0.35 for S2

        corrected = b02 - k * glint
        return np.clip(corrected, 0, 1.0).astype(np.float32)

    def acolite_correct(self, input_path: str, output_dir: str,
                        sunglint: bool = True) -> str:
        """
        Run ACOLITE atmospheric correction via CLI.

        Supports sensors that ACOLITE can handle:
          - Sentinel-2 (--sensor S2)
          - Pléiades (--sensor PLEIADES)
          - SPOT (--sensor SPOT)
          - WorldView (--sensor WV3)

        Returns path to output directory with corrected products.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "acolite_cli",
            "--input", str(input_path),
            "--output", str(output_path),
            "--product", "BOA",
            "--sensor", self.sensor.acolite_sensor,
            "--proc", "water",
            "--sunglint", str(sunglint).lower(),
            "--aot-method", "image",
            "--output-format", "GeoTIFF",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                warnings.warn(f"ACOLITE failed (exit {result.returncode}): {result.stderr[:500]}")
                return ""
            return str(output_path)
        except FileNotFoundError:
            warnings.warn("ACOLITE CLI not found. Install: pip install acolite")
            return ""
        except subprocess.TimeoutExpired:
            warnings.warn("ACOLITE timed out after 600s")
            return ""

    def _get_path_radiance(self, wavelength_nm: int) -> float:
        """Get atmospheric path radiance for a wavelength (interpolate if needed)."""
        wavelengths = sorted(self.PATH_RADIANCE.keys())

        # Exact match
        if wavelength_nm in self.PATH_RADIANCE:
            return self.PATH_RADIANCE[wavelength_nm]

        # Linear interpolation
        for i in range(len(wavelengths) - 1):
            if wavelengths[i] <= wavelength_nm <= wavelengths[i + 1]:
                w1, w2 = wavelengths[i], wavelengths[i + 1]
                l1, l2 = self.PATH_RADIANCE[w1], self.PATH_RADIANCE[w2]
                t = (wavelength_nm - w1) / (w2 - w1)
                return l1 + t * (l2 - l1)

        # Extrapolate
        if wavelength_nm < wavelengths[0]:
            return self.PATH_RADIANCE[wavelengths[0]]
        return self.PATH_RADIANCE[wavelengths[-1]]

    def _optical_depth(self, wavelength_nm: int) -> float:
        """
        Estimate aerosol + molecular optical depth for coastal Algarve.

        Uses Angstrom law: tau = beta * lambda^(-alpha)
        Where alpha ~1.3 (maritime aerosols), beta ~0.1 (low turbidity)
        """
        beta = 0.1   # Turbidity coefficient (Algarve: low-moderate)
        alpha = 1.3  # Angstrom exponent (maritime)
        lam_um = wavelength_nm / 1000.0
        return beta * (lam_um ** (-alpha))


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════→═══════════════

def correct_b02(dn_array: np.ndarray, sensor_name: str = "sentinel-2") -> np.ndarray:
    """Quick B02 correction — the only band that matters for reef ID."""
    ac = AtmosphericCorrector(sensor_name)
    return ac.dn_to_reflectance(dn_array, "B02")


def correct_all_bands(dn_dict: Dict[str, np.ndarray], sensor_name: str = "sentinel-2") -> Dict[str, np.ndarray]:
    """Correct all bands from DN to BOA reflectance."""
    ac = AtmosphericCorrector(sensor_name)
    return {band: ac.dn_to_reflectance(arr, band) for band, arr in dn_dict.items()}


def run_acolite_for_sensor(input_path: str, output_dir: str, sensor_name: str = "sentinel-2") -> str:
    """Run ACOLITE correction for any supported sensor."""
    ac = AtmosphericCorrector(sensor_name)
    return ac.acolite_correct(input_path, output_dir)
