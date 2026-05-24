#!/usr/bin/env python3
"""
ih_bathy_features.py — IH/DGRM Bathymetry Feature Engineering
===============================================================

Integrates the official ArcGIS REST service from DGRM/IH into the reef
imagery pipeline as reusable bathymetry-derived features for ML training
and operational prediction.

Data source
-----------
ArcGIS REST: https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/
              Dados_entidades_externas/Batimetrica_IH/MapServer/0
Layer: "Isobatimetricas, Escala 1:150.000 (Fonte: IH)"
Geometry: polyline  |  CRS served: EPSG:4326  |  Source CRS: EPSG:3763
Key attribute: Depth (metres)

What this module does
---------------------
1.  Download isobath polylines for any AOI using bbox chunking (handles
    maxRecordCount=1000 by tiling large areas into smaller requests).
2.  Cache results locally in GeoPackage for fast reuse.
3.  Reproject to EPSG:3763 (PT-TM06 / ETRS89-TM06) for accurate metric
    calculations.
4.  Build bathymetry-derived feature vectors:
        nearest_isobath_distance_m
        nearest_isobath_depth_m
        dist_to_isobath_10m, 20m, 30m, 50m, 100m
        bathymetry_zone_class
        bathymetry_slope_proxy
        contour_density_proxy
5.  Integrate seamlessly with existing training & inference paths.

Usage (standalone example)
--------------------------
    from src.ih_bathy_features import BathyFeatureEngine
    engine = BathyFeatureEngine(cache_dir="data/cache")
    features = engine.compute_features_for_point(lon=-8.21, lat=37.06)

Author: 3ruiruirui-sketch
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import requests

log = logging.getLogger(__name__)

# ── Service constants ──────────────────────────────────────────────────────────
_IH_BASE = (
    "https://webgis.dgrm.mm.gov.pt/arcgis/rest/services/"
    "Dados_entidades_externas/Batimetrica_IH/MapServer/0"
)
_QUERY_URL = f"{_IH_BASE}/query"

# All isobath depths available in the layer (verified from service metadata)
ALL_ISOBATHS = [0, 2, 10, 20, 30, 50, 100, 200, 400, 500, 1000, 2000, 3000, 4000]
# Subset used for reef-analysis feature engineering
REEF_ISOBATHS = [10, 20, 30, 50, 100]

# Approximate metres per degree at Algarve latitudes (~37°N)
M_PER_DEG = 111_320.0

# Bbox chunk size (degrees).  Service maxRecordCount=1000, so we use
# modest tiles to stay well under the limit even in contour-dense areas.
DEFAULT_CHUNK_DEG = 0.10  # ~11 km per tile

# Retries / back-off for transient ArcGIS failures
_MAX_RETRIES = 3
_RETRY_DELAY_S = 2.0


# =============================================================================
# A.  Downloader  (bbox chunking + merge + dedup)
# =============================================================================

class IHBathyDownloader:
    """Chunked downloader for DGRM/IH isobath polylines with local cache."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        chunk_deg: float = DEFAULT_CHUNK_DEG,
        timeout: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_deg = chunk_deg
        self.timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_for_aoi(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        depths: list[int] | None = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Fetch all isobath polylines for an AOI, using cache when available.

        Returns a deduplicated list of feature dicts:
            {
                "depth": float,
                "coords": list[list[float]],   # [[lon,lat], ...]
                "shape_leng": float,
                "objectid": int,
            }
        """
        depths = depths or ALL_ISOBATHS
        cache_path = self._cache_path(min_lon, min_lat, max_lon, max_lat, depths)

        if use_cache and cache_path.exists():
            log.info("IH bathy cache hit: %s", cache_path)
            return self._load_cache(cache_path)

        log.info(
            "IH bathy fetching AOI [%.4f,%.4f → %.4f,%.4f] (chunk=%.2f°)",
            min_lon, min_lat, max_lon, max_lat, self.chunk_deg,
        )
        features = self._fetch_chunked(min_lon, min_lat, max_lon, max_lat, depths)
        features = self._deduplicate(features)

        if use_cache:
            self._save_cache(cache_path, features)

        log.info("IH bathy AOI complete: %d unique polylines", len(features))
        return features

    def clear_cache(self) -> None:
        """Remove all cached GeoPackage files."""
        for p in self.cache_dir.glob("ih_bathy_*.gpkg"):
            p.unlink()
        log.info("IH bathy cache cleared")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_chunked(
        self,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
        depths: list[int],
    ) -> list[dict]:
        """Tile AOI into chunks, fetch each, merge results."""
        tiles = self._tile_bbox(min_lon, min_lat, max_lon, max_lat, self.chunk_deg)
        all_features: list[dict] = []

        for i, (w, s, e, n) in enumerate(tiles, 1):
            chunk = self._fetch_single_bbox(w, s, e, n, depths)
            all_features.extend(chunk)
            log.debug("  Tile %d/%d: %d features", i, len(tiles), len(chunk))

        return all_features

    def _fetch_single_bbox(
        self,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
        depths: list[int],
    ) -> list[dict]:
        """One ArcGIS REST query with retry logic."""
        depth_filter = ", ".join(str(d) for d in depths)
        params = {
            "where": f"Depth IN ({depth_filter})",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "OBJECTID,Depth,Shape_Leng",
            "returnGeometry": "true",
            "f": "json",
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(_QUERY_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    log.warning("IH service error (attempt %d): %s", attempt, data["error"])
                    time.sleep(_RETRY_DELAY_S * attempt)
                    continue

                return self._parse_features(data)

            except requests.RequestException as exc:
                log.warning("IH fetch failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_S * attempt)

        log.error("IH fetch exhausted retries for bbox [%.4f,%.4f,%.4f,%.4f]",
                  min_lon, min_lat, max_lon, max_lat)
        return []

    @staticmethod
    def _parse_features(data: dict) -> list[dict]:
        """Parse ArcGIS JSON response into uniform feature dicts."""
        out = []
        for feat in data.get("features", []):
            attrs = feat.get("attributes", {})
            geom = feat.get("geometry", {})
            paths = geom.get("paths", [])
            for path in paths:
                out.append({
                    "depth": float(attrs.get("Depth", 0)),
                    "coords": path,  # [[lon, lat], ...]
                    "shape_leng": float(attrs.get("Shape_Leng", 0.0)),
                    "objectid": int(attrs.get("OBJECTID", 0)),
                })
        return out

    @staticmethod
    def _deduplicate(features: list[dict]) -> list[dict]:
        """Remove exact duplicate polylines by (objectid, depth, first coord)."""
        seen: set[str] = set()
        uniq: list[dict] = []
        for f in features:
            key = f"{f['objectid']}:{f['depth']:.0f}:{f['coords'][0][0]:.6f}:{f['coords'][0][1]:.6f}"
            if key not in seen:
                seen.add(key)
                uniq.append(f)
        if len(uniq) < len(features):
            log.info("IH bathy dedup: removed %d duplicates", len(features) - len(uniq))
        return uniq

    # ── Cache helpers ───────────────────────────────────────────────────────

    def _cache_path(
        self,
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
        depths: list[int],
    ) -> Path:
        """Deterministic cache filename from bbox + depth list."""
        bbox_str = f"{min_lon:.5f}_{min_lat:.5f}_{max_lon:.5f}_{max_lat:.5f}"
        depth_str = "_".join(str(d) for d in sorted(depths))
        sig = hashlib.sha256(f"{bbox_str}:{depth_str}".encode()).hexdigest()[:16]
        return self.cache_dir / f"ih_bathy_{sig}.gpkg"

    def _save_cache(self, path: Path, features: list[dict]) -> None:
        """Save to GeoPackage via geopandas (fallback to JSON if geopandas missing)."""
        try:
            import geopandas as gpd
            from shapely.geometry import LineString

            records = []
            for f in features:
                records.append({
                    "geometry": LineString(f["coords"]),
                    "depth": f["depth"],
                    "shape_leng": f["shape_leng"],
                    "objectid": f["objectid"],
                })
            gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
            gdf.to_file(path, driver="GPKG")
            log.debug("Cache saved (GeoPackage): %s (%d features)", path, len(features))
        except Exception as exc:
            log.warning("GeoPackage cache failed (%s), falling back to JSON", exc)
            json_path = path.with_suffix(".json")
            with open(json_path, "w") as fh:
                json.dump(features, fh)
            log.debug("Cache saved (JSON): %s", json_path)

    def _load_cache(self, path: Path) -> list[dict]:
        """Load from GeoPackage or JSON fallback."""
        if path.suffix == ".json" or not path.exists():
            json_path = path.with_suffix(".json")
            if json_path.exists():
                with open(json_path) as fh:
                    return json.load(fh)
            return []

        try:
            import geopandas as gpd
            gdf = gpd.read_file(path)
            return [
                {
                    "depth": float(row["depth"]),
                    "coords": list(row.geometry.coords),
                    "shape_leng": float(row.get("shape_leng", 0.0)),
                    "objectid": int(row.get("objectid", 0)),
                }
                for _, row in gdf.iterrows()
            ]
        except Exception as exc:
            log.warning("Cache load failed (%s), returning empty", exc)
            return []

    @staticmethod
    def _tile_bbox(
        min_lon: float, min_lat: float,
        max_lon: float, max_lat: float,
        step: float,
    ) -> list[tuple[float, float, float, float]]:
        """Generate non-overlapping tile bboxes."""
        tiles = []
        lat = min_lat
        while lat < max_lat:
            lon = min_lon
            while lon < max_lon:
                tiles.append((
                    lon, lat,
                    min(lon + step, max_lon),
                    min(lat + step, max_lat),
                ))
                lon += step
            lat += step
        return tiles


# =============================================================================
# B.  Feature Engineering
# =============================================================================

class BathyFeatureEngine:
    """Compute bathymetry-derived features for points or grids."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        downloader: IHBathyDownloader | None = None,
    ):
        self.downloader = downloader or IHBathyDownloader(cache_dir=cache_dir)
        self._features_cache: dict[str, list[dict]] = {}  # bbox_key → features

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_features_for_point(
        self,
        lon: float,
        lat: float,
        buffer_m: float = 5_000.0,
        depths: list[int] | None = None,
    ) -> dict[str, float | str | None]:
        """
        Compute a full feature vector for a single (lon, lat) point.

        Features returned:
            nearest_isobath_distance_m   — distance to closest isobath
            nearest_isobath_depth_m      — depth label of that isobath
            dist_to_isobath_10m          — distance to 10m isobath (or inf)
            dist_to_isobath_20m
            dist_to_isobath_30m
            dist_to_isobath_50m
            dist_to_isobath_100m
            bathymetry_zone_class        — categorical zone
            bathymetry_slope_proxy       — proxy for local gradient
            contour_density_proxy        — total contour length / AOI area
        """
        depths = depths or REEF_ISOBATHS
        features = self._fetch_for_point(lon, lat, buffer_m, depths)

        if not features:
            return self._empty_features()

        # All metric distances computed in EPSG:3763 for accuracy
        dists_3763 = self._compute_distances_3763(lon, lat, features)

        # Nearest overall
        nearest_depth, nearest_dist = min(
            ((d, dists_3763.get(f"dist_{int(d)}m", np.inf)) for d in depths),
            key=lambda x: x[1],
        )

        # Zone classification (same rules as bathy_calibrator but using metric dists)
        zone = self._classify_zone(dists_3763)

        # Slope proxy: std of depth values among nearby isobaths
        slope_proxy = self._slope_proxy(features, lon, lat)

        # Contour density: total line length (m) / AOI area (km²)
        aoi_km2 = (buffer_m * 2 / 1000) ** 2
        total_length = sum(f.get("shape_leng", 0.0) for f in features)
        density_proxy = total_length / aoi_km2 if aoi_km2 > 0 else 0.0

        return {
            "nearest_isobath_distance_m": round(float(nearest_dist), 1),
            "nearest_isobath_depth_m": float(nearest_depth),
            "dist_to_isobath_10m": round(dists_3763.get("dist_10m", np.inf), 1),
            "dist_to_isobath_20m": round(dists_3763.get("dist_20m", np.inf), 1),
            "dist_to_isobath_30m": round(dists_3763.get("dist_30m", np.inf), 1),
            "dist_to_isobath_50m": round(dists_3763.get("dist_50m", np.inf), 1),
            "dist_to_isobath_100m": round(dists_3763.get("dist_100m", np.inf), 1),
            "bathymetry_zone_class": zone,
            "bathymetry_slope_proxy": round(slope_proxy, 4),
            "contour_density_proxy": round(density_proxy, 2),
            "n_isobaths_in_aoi": len(features),
        }

    def compute_features_for_points(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        buffer_m: float = 5_000.0,
        depths: list[int] | None = None,
    ) -> list[dict[str, float | str | None]]:
        """Batch version for arrays of coordinates (e.g. training set)."""
        # Fetch once for the envelope of all points
        min_lon, max_lon = float(lons.min()), float(lons.max())
        min_lat, max_lat = float(lats.min()), float(lats.max())
        # Expand by buffer
        deg_buf = buffer_m / M_PER_DEG
        features = self._fetch_for_bbox(
            min_lon - deg_buf, min_lat - deg_buf,
            max_lon + deg_buf, max_lat + deg_buf,
            depths,
        )
        # Compute per-point (could be vectorised with scipy cKDTree in future)
        return [
            self.compute_features_for_point(lon, lat, buffer_m=buffer_m, depths=depths)
            for lon, lat in zip(lons, lats)
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_for_point(
        self, lon: float, lat: float, buffer_m: float, depths: list[int]
    ) -> list[dict]:
        """Fetch isobaths for a point + buffer, with in-memory caching."""
        deg_buf = buffer_m / M_PER_DEG
        bbox = (lon - deg_buf, lat - deg_buf, lon + deg_buf, lat + deg_buf)
        key = f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"

        if key not in self._features_cache:
            self._features_cache[key] = self.downloader.fetch_for_aoi(
                *bbox, depths=depths
            )
        return self._features_cache[key]

    def _fetch_for_bbox(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
        depths: list[int] | None,
    ) -> list[dict]:
        key = f"{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}"
        if key not in self._features_cache:
            self._features_cache[key] = self.downloader.fetch_for_aoi(
                min_lon, min_lat, max_lon, max_lat, depths=depths
            )
        return self._features_cache[key]

    @staticmethod
    def _compute_distances_3763(
        lon: float, lat: float, features: list[dict]
    ) -> dict[str, float]:
        """Compute metric distances in EPSG:3763 (PT-TM06) for accuracy."""
        try:
            from pyproj import Transformer
            # EPSG:4326 → EPSG:3763 (PT-TM06 / ETRS89-TM06)
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
            px, py = transformer.transform(lon, lat)
        except Exception:
            # Fallback: haversine approximation
            return BathyFeatureEngine._compute_distances_haversine(lon, lat, features)

        dists: dict[str, float] = {}
        for target_depth in REEF_ISOBATHS:
            min_d = np.inf
            for feat in features:
                if feat["depth"] != float(target_depth):
                    continue
                for node in feat["coords"]:
                    nx, ny = transformer.transform(node[0], node[1])
                    d = float(np.hypot(px - nx, py - ny))
                    if d < min_d:
                        min_d = d
            dists[f"dist_{target_depth}m"] = min_d if min_d < np.inf else np.inf
        return dists

    @staticmethod
    def _compute_distances_haversine(
        lon: float, lat: float, features: list[dict]
    ) -> dict[str, float]:
        """Fallback when pyproj unavailable — haversine in metres."""
        R = 6_371_000.0
        phi1, lam1 = np.radians(lat), np.radians(lon)
        dists: dict[str, float] = {}

        for target_depth in REEF_ISOBATHS:
            min_d = np.inf
            for feat in features:
                if feat["depth"] != float(target_depth):
                    continue
                for node in feat["coords"]:
                    phi2, lam2 = np.radians(node[1]), np.radians(node[0])
                    dphi = phi2 - phi1
                    dlam = lam2 - lam1
                    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
                    d = R * 2 * np.arcsin(np.sqrt(a))
                    if d < min_d:
                        min_d = d
            dists[f"dist_{target_depth}m"] = min_d if min_d < np.inf else np.inf
        return dists

    @staticmethod
    def _classify_zone(dists: dict[str, float]) -> str:
        """Classify depth zone from metric distances (same logic as bathy_calibrator)."""
        d10 = dists.get("dist_10m", np.inf)
        d20 = dists.get("dist_20m", np.inf)
        d30 = dists.get("dist_30m", np.inf)
        d50 = dists.get("dist_50m", np.inf)

        available = {}
        if not np.isinf(d10):
            available["10m"] = d10
        if not np.isinf(d20):
            available["20m"] = d20
        if not np.isinf(d30):
            available["30m"] = d30
        if not np.isinf(d50):
            available["50m"] = d50

        if d10 < 200:
            return "very_shallow"
        elif d20 < 500:
            return "shallow_reef"
        elif d10 < 1500 or d20 < 1500:
            return "nearshore_mid"
        elif not np.isinf(d30) and d30 < 1000:
            return "mid_depth"
        elif not np.isinf(d50) and d50 < 500:
            return "offshore"
        else:
            min_known = min(available.values()) if available else np.inf
            return "nearshore_mid" if min_known < 2000 else "offshore"

    @staticmethod
    def _slope_proxy(features: list[dict], lon: float, lat: float) -> float:
        """Proxy for local bathymetric slope: std of depths among nearby contours."""
        # Gather depths of all contours within ~2km of the point
        nearby_depths = []
        for feat in features:
            # Check if any vertex is within ~2km (rough)
            for node in feat["coords"][:5]:  # sample first 5 vertices
                d = float(np.hypot((node[0] - lon) * M_PER_DEG * np.cos(np.radians(lat)),
                                   (node[1] - lat) * M_PER_DEG))
                if d < 2000:
                    nearby_depths.append(feat["depth"])
                    break
        if len(nearby_depths) < 2:
            return 0.0
        return float(np.std(nearby_depths))

    @staticmethod
    def _empty_features() -> dict[str, float | str | None]:
        return {
            "nearest_isobath_distance_m": np.inf,
            "nearest_isobath_depth_m": None,
            "dist_to_isobath_10m": np.inf,
            "dist_to_isobath_20m": np.inf,
            "dist_to_isobath_30m": np.inf,
            "dist_to_isobath_50m": np.inf,
            "dist_to_isobath_100m": np.inf,
            "bathymetry_zone_class": "unknown",
            "bathymetry_slope_proxy": 0.0,
            "contour_density_proxy": 0.0,
            "n_isobaths_in_aoi": 0,
        }


# =============================================================================
# C.  Convenience function for pipeline integration
# =============================================================================

def get_bathy_features_for_summary(
    lon: float,
    lat: float,
    cache_dir: str | Path = "data/cache",
    buffer_m: float = 5_000.0,
) -> dict[str, float | str | None]:
    """
    One-liner used by reef_ml_predictor_acolite to append bathymetry
    columns to the summary CSV.
    """
    engine = BathyFeatureEngine(cache_dir=cache_dir)
    return engine.compute_features_for_point(lon, lat, buffer_m=buffer_m)


# =============================================================================
# D.  CLI demo
# =============================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="IH Bathymetry Feature Engineering Demo")
    p.add_argument("--lon", type=float, default=-8.210492)
    p.add_argument("--lat", type=float, default=37.069071)
    p.add_argument("--buffer-m", type=float, default=5_000.0)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--clear-cache", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = BathyFeatureEngine(cache_dir=args.cache_dir)

    if args.clear_cache:
        engine.downloader.clear_cache()

    print(f"\nComputing bathymetry features for ({args.lon}, {args.lat}) ...\n")
    feats = engine.compute_features_for_point(args.lon, args.lat, buffer_m=args.buffer_m)

    print(json.dumps(feats, indent=2, default=str))
