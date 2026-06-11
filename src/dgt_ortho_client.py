"""
DGT orthophoto client — STAC-based COG downloader for ORTOSAT-2023 and
historical air orthos (1995–2021).

Why this exists
---------------
The v3 pipeline uses DGT WMS/WCS to retrieve ORTOSAT-2023 30 cm imagery.
WMS is unreliable for programmatic use (SharePoint redirect, session cookies,
max-image-size limits).  The DGT STAC API at dgt-be.a.incd.pt:8081 exposes
the same data as Cloud-Optimized GeoTIFFs hosted on a MinIO bucket at
stor-002.a.acnca.pt, accessible via the DGT Centro de Dados portal.

Authentication — how it actually works
--------------------------------------
There are NO public S3 keys.  The correct access path is:

  1. Register a free account at https://cdd.dgterritorio.gov.pt
     (open self-registration; fields: username, email, firstName,
     lastName, organizationname, password; license CC-BY 4.0)

  2. Set environment variables:
       DGT_CDD_USERNAME=your_username
       DGT_CDD_PASSWORD=your_password

  3. This client uses the Keycloak OIDC password grant to obtain a
     JWT bearer token from auth.cdd.dgterritorio.gov.pt, then
     calls the backend proxy at cdd.dgterritorio.gov.pt/dgt-be/v1/
     which signs and proxies the MinIO downloads.

  STAC metadata queries (list_tiles, best_available) remain publicly
  accessible without credentials.  Download calls log a clear error
  and return [] when credentials are absent rather than raising.

Available collections
---------------------
  ORTOSAT-2023  — 30 cm RGBI COG (satellite, RGB + Near-Infrared)
  ORTOS-2021    — 25 cm RGBI COG (aerial, RGB + Near-Infrared)
  ORTOS-2018    — 25 cm RGBI COG
  ORTOS-2015    — 25 cm RGBI COG
  ORTOS-2012    — 25 cm RGBI COG
  ORTOS-2010    — 25 cm RGBI COG
  ORTOS-2007    — 25 cm RGBI COG
  ORTOS-2004    — 25 cm RGBI COG
  ORTOS-1995    — aerial false-colour infrared film scan (RGBI equivalent)

Usage
-----
    from src.dgt_ortho_client import DGTOrthoClient

    client = DGTOrthoClient(bbox=(-8.25, 37.04, -8.17, 37.10),
                            output_dir="./outputs/ortho")

    # Download ORTOSAT-2023 tiles for bbox
    paths = client.download_tiles("ORTOSAT-2023")

    # Download a specific historical year
    paths = client.download_tiles("ORTOS-2018")

    # Compute NDVI from the NIR (band 4) and Red (band 3) channels
    ndvi_path = client.compute_ndvi(paths)

    # Quick change-detection summary between two years
    report = client.change_summary("ORTOS-2015", "ORTOSAT-2023")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from src.dgt_cdd_auth import get_cdd_session, get_signed_url, invalidate

logger = logging.getLogger(__name__)

# Module-level aliases kept for test patching compatibility
_DGT_USER = os.environ.get("DGT_CDD_USERNAME")
_DGT_PASS = os.environ.get("DGT_CDD_PASSWORD")


def _build_authenticated_session() -> Optional[requests.Session]:
    """Compatibility shim — delegates to dgt_cdd_auth.get_cdd_session."""
    return get_cdd_session(force_refresh=True)


def _get_session() -> Optional[requests.Session]:
    return get_cdd_session()


def _invalidate_session() -> None:
    invalidate()

# ── STAC catalogue constants ───────────────────────────────────────────────────

DGT_STAC_ROOT = "https://dgt-be.a.incd.pt:8081"

# Collection IDs available on the DGT STAC (all cover Portugal mainland)
ORTHO_COLLECTIONS: Dict[str, str] = {
    "ORTOSAT-2023": "ORTOSAT-2023",   # 30 cm satellite RGBI
    "ORTOS-2021":   "ORTOS-2021",     # 25 cm aerial RGBI
    "ORTOS-2018":   "ORTOS-2018",
    "ORTOS-2015":   "ORTOS-2015",
    "ORTOS-2012":   "ORTOS-2012",
    "ORTOS-2010":   "ORTOS-2010",
    "ORTOS-2007":   "ORTOS-2007",
    "ORTOS-2004":   "ORTOS-2004",
    "ORTOS-1995":   "ORTOS-1995",
}

# Sorted from newest to oldest for change-detection helpers
COLLECTIONS_BY_YEAR: List[str] = [
    "ORTOSAT-2023", "ORTOS-2021", "ORTOS-2018", "ORTOS-2015",
    "ORTOS-2012", "ORTOS-2010", "ORTOS-2007", "ORTOS-2004", "ORTOS-1995",
]

# Band order in DGT COGs: Red=1, Green=2, Blue=3, NIR=4
BAND_RED = 1
BAND_GREEN = 2
BAND_BLUE = 3
BAND_NIR = 4

try:
    import rasterio
    from rasterio.merge import merge as _rio_merge
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class DGTOrthoClient:
    """Download DGT orthophoto COGs and compute spectral indices."""

    def __init__(self,
                 bbox: Tuple[float, float, float, float],
                 output_dir: str = "./outputs/ortho",
                 cache: bool = True) -> None:
        """
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 (lon, lat)
            output_dir: where to save downloaded tiles and derived products
            cache: if True, skip files that already exist locally
        """
        self.bbox = bbox
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache

    # ── STAC discovery ─────────────────────────────────────────────────────────

    def list_tiles(self, collection: str, limit: int = 100) -> List[Dict]:
        """Query the DGT STAC and return items that intersect the bbox.

        Args:
            collection: one of the keys in ORTHO_COLLECTIONS
            limit: maximum number of items to return per request

        Returns:
            list of STAC feature dicts
        """
        coll_id = ORTHO_COLLECTIONS.get(collection, collection)
        url = f"{DGT_STAC_ROOT}/collections/{coll_id}/items"
        params = {
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "limit": limit,
            "f": "json",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            fc = r.json()
            n = fc.get("context", {}).get("returned", 0)
            logger.info(f"{collection}: {n} tiles in bbox")
            return fc.get("features", [])
        except requests.RequestException as exc:
            logger.error(f"STAC query failed for {collection}: {exc}")
            return []

    # ── Download ───────────────────────────────────────────────────────────────

    def _signed_download_url(self, collection: str, item_id: str) -> Optional[str]:
        """Return a signed download URL via the shared dgt_cdd_auth helper."""
        return get_signed_url(collection, item_id)

    def download_tiles(self, collection: str, limit: int = 100) -> List[Path]:
        """Download all COG tiles for a collection that overlap the bbox.

        Requires a free DGT CDD account (register at https://cdd.dgterritorio.gov.pt).
        Set DGT_CDD_USERNAME and DGT_CDD_PASSWORD environment variables.

        Files are saved to ``output_dir/collection/``.  Already-downloaded
        files are skipped when ``cache=True``.

        Args:
            collection: collection name (e.g. "ORTOSAT-2023", "ORTOS-2018")
            limit: max tiles to fetch

        Returns:
            list of local file paths
        """
        tiles_dir = self.output_dir / collection
        tiles_dir.mkdir(exist_ok=True)

        features = self.list_tiles(collection, limit=limit)
        if not features:
            logger.warning(f"No tiles found for {collection}")
            return []

        paths: List[Path] = []
        for feat in features:
            item_id  = feat.get("id", "")
            coll_id  = ORTHO_COLLECTIONS.get(collection, collection)
            asset_key = "visual" if "visual" in feat.get("assets", {}) else "Data"
            raw_href  = feat.get("assets", {}).get(asset_key, {}).get("href", "")
            fname = Path(raw_href).name or f"{item_id}.tif"
            local = tiles_dir / fname

            if local.exists() and self.cache:
                logger.debug(f"Cached: {local.name}")
                paths.append(local)
                continue

            logger.info(f"Downloading {collection}/{fname}")
            if not _DGT_USER or not _DGT_PASS:
                logger.error(
                    "DGT_CDD_USERNAME / DGT_CDD_PASSWORD not set. "
                    "Register free at https://cdd.dgterritorio.gov.pt then set env vars."
                )
                continue

            signed_url = self._signed_download_url(coll_id, item_id)
            if signed_url is None:
                logger.error(
                    "Could not obtain a signed download URL for %s/%s — "
                    "check credentials.", collection, item_id
                )
                continue

            session = _get_session()
            try:
                with session.get(signed_url, stream=True, timeout=120) as resp:
                    if resp.status_code in (401, 403):
                        logger.error(
                            "  HTTP %d for %s — session expired or account lacks access. "
                            "Re-check credentials at cdd.dgterritorio.gov.pt.",
                            resp.status_code, fname,
                        )
                        _invalidate_session()
                        continue
                    resp.raise_for_status()
                    with open(local, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk)
                size_mb = local.stat().st_size / 1e6
                logger.info(f"  Saved {local.name} ({size_mb:.1f} MB)")
                paths.append(local)
            except Exception as exc:
                logger.error(f"  Failed {fname}: {exc}")

        return paths

    # ── Mosaic ─────────────────────────────────────────────────────────────────

    def mosaic(self, tile_paths: List[Path], output_stem: str = "mosaic") -> Optional[Path]:
        """Merge downloaded COG tiles into a single GeoTIFF.

        Args:
            tile_paths: local paths returned by download_tiles()
            output_stem: base name for the output file

        Returns:
            path to the merged GeoTIFF, or None on failure
        """
        if not HAS_RASTERIO:
            logger.error("rasterio is required for mosaicing")
            return None
        if not tile_paths:
            return None

        out_path = self.output_dir / f"{output_stem}_mosaic.tif"
        if out_path.exists() and self.cache:
            return out_path

        try:
            with contextlib.ExitStack() as stack:
                src_files = [stack.enter_context(rasterio.open(str(p))) for p in tile_paths]
                arr, transform = _rio_merge(src_files)
                meta = src_files[0].meta.copy()
                meta.update(height=arr.shape[1], width=arr.shape[2],
                            transform=transform)
            with rasterio.open(str(out_path), "w", **meta) as dst:
                dst.write(arr)
            logger.info(f"Mosaic saved: {out_path}")
            return out_path
        except Exception as exc:
            logger.error(f"Mosaic failed: {exc}")
            return None

    # ── Spectral indices ───────────────────────────────────────────────────────

    def compute_ndvi(self,
                     tile_paths: List[Path],
                     output_stem: str = "ndvi") -> Optional[Path]:
        """Compute NDVI = (NIR − Red) / (NIR + Red) from RGBI COG tiles.

        DGT COGs carry 4 bands: R=1, G=2, B=3, NIR=4.
        Output is a Float32 GeoTIFF with values in [−1, 1]; nodata = NaN.

        Args:
            tile_paths: local tile paths (typically from download_tiles)
            output_stem: base name for the NDVI output file

        Returns:
            path to NDVI GeoTIFF, or None on failure
        """
        if not HAS_RASTERIO:
            logger.error("rasterio required for NDVI")
            return None

        out_path = self.output_dir / f"{output_stem}_ndvi.tif"
        if out_path.exists() and self.cache:
            return out_path

        try:
            # Mosaic first if multiple tiles
            if len(tile_paths) > 1:
                mosaic_path = self.mosaic(tile_paths, output_stem=output_stem)
                if mosaic_path is None:
                    return None
                src_path = mosaic_path
            else:
                src_path = tile_paths[0]

            with rasterio.open(str(src_path)) as src:
                if src.count < BAND_NIR:
                    logger.error(
                        f"{src_path.name} has only {src.count} bands; NIR (band 4) required")
                    return None
                red = src.read(BAND_RED).astype(np.float32)
                nir = src.read(BAND_NIR).astype(np.float32)
                nodata = src.nodata
                profile = src.profile.copy()

            if nodata is not None:
                red = np.where(red == nodata, np.nan, red)
                nir = np.where(nir == nodata, np.nan, nir)

            denom = nir + red
            ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)

            profile.update(count=1, dtype="float32", nodata=np.nan)
            with rasterio.open(str(out_path), "w", **profile) as dst:
                dst.write(ndvi.astype("float32"), 1)

            valid = ndvi[~np.isnan(ndvi)]
            logger.info(
                f"NDVI saved: {out_path}  "
                f"mean={np.mean(valid):.3f}  min={np.min(valid):.3f}  max={np.max(valid):.3f}")
            return out_path

        except Exception as exc:
            logger.error(f"NDVI computation failed: {exc}")
            return None

    def compute_ndwi(self,
                     tile_paths: List[Path],
                     output_stem: str = "ndwi") -> Optional[Path]:
        """Compute NDWI = (Green − NIR) / (Green + NIR).

        Positive NDWI indicates open water; useful for delineating the
        reef–water boundary and detecting turbid plumes.
        """
        if not HAS_RASTERIO:
            logger.error("rasterio required for NDWI")
            return None

        out_path = self.output_dir / f"{output_stem}_ndwi.tif"
        if out_path.exists() and self.cache:
            return out_path

        try:
            if len(tile_paths) > 1:
                mosaic_path = self.mosaic(tile_paths, output_stem=output_stem)
                if mosaic_path is None:
                    return None
                src_path = mosaic_path
            else:
                src_path = tile_paths[0]

            with rasterio.open(str(src_path)) as src:
                if src.count < BAND_NIR:
                    logger.error(f"Need ≥4 bands for NDWI; got {src.count}")
                    return None
                green = src.read(BAND_GREEN).astype(np.float32)
                nir = src.read(BAND_NIR).astype(np.float32)
                nodata = src.nodata
                profile = src.profile.copy()

            if nodata is not None:
                green = np.where(green == nodata, np.nan, green)
                nir = np.where(nir == nodata, np.nan, nir)

            denom = green + nir
            ndwi = np.where(denom == 0, np.nan, (green - nir) / denom)

            profile.update(count=1, dtype="float32", nodata=np.nan)
            with rasterio.open(str(out_path), "w", **profile) as dst:
                dst.write(ndwi.astype("float32"), 1)

            logger.info(f"NDWI saved: {out_path}")
            return out_path

        except Exception as exc:
            logger.error(f"NDWI failed: {exc}")
            return None

    # ── Change detection ───────────────────────────────────────────────────────

    def change_summary(self,
                       collection_early: str,
                       collection_late: str) -> Dict:
        """Download NDVI for two years and compute a pixel-level change report.

        Returns a dict with mean/std ΔNDVI and the fraction of pixels that show
        significant greening (ΔNDVI > +0.05) or browning (ΔNDVI < −0.05).
        Useful for detecting coastal vegetation loss / reef habitat shifts.

        Args:
            collection_early: older collection, e.g. "ORTOS-2015"
            collection_late: newer collection, e.g. "ORTOSAT-2023"

        Returns:
            dict with keys: delta_ndvi_mean, delta_ndvi_std,
                            frac_greening, frac_browning, n_valid_pixels
        """
        if not HAS_RASTERIO:
            return {"error": "rasterio not installed"}

        early_tiles = self.download_tiles(collection_early)
        late_tiles = self.download_tiles(collection_late)

        if not early_tiles or not late_tiles:
            return {"error": f"Could not download tiles for {collection_early} or {collection_late}"}

        ndvi_early_path = self.compute_ndvi(early_tiles, output_stem=collection_early)
        ndvi_late_path = self.compute_ndvi(late_tiles, output_stem=collection_late)

        if ndvi_early_path is None or ndvi_late_path is None:
            return {"error": "NDVI computation failed for one or both collections"}

        try:
            with rasterio.open(str(ndvi_early_path)) as src:
                ndvi_e = src.read(1).astype(np.float32)
            with rasterio.open(str(ndvi_late_path)) as src_l:
                # Reproject late to match early grid if needed
                with rasterio.open(str(ndvi_early_path)) as _chk:
                    _early_shape = _chk.shape
                if src_l.shape != _early_shape:
                    import rioxarray as rxr
                    late_da = rxr.open_rasterio(str(ndvi_late_path), masked=True)
                    early_da = rxr.open_rasterio(str(ndvi_early_path), masked=True)
                    late_da = late_da.rio.reproject_match(early_da)
                    ndvi_l = late_da.values[0].astype(np.float32)
                else:
                    ndvi_l = src_l.read(1).astype(np.float32)

            delta = ndvi_l - ndvi_e
            valid = ~(np.isnan(delta))
            n_valid = int(valid.sum())
            if n_valid == 0:
                return {"error": "No valid overlapping pixels"}

            d = delta[valid]
            return {
                "collection_early": collection_early,
                "collection_late": collection_late,
                "delta_ndvi_mean": float(np.mean(d)),
                "delta_ndvi_std": float(np.std(d)),
                "frac_greening": float((d > 0.05).sum() / n_valid),
                "frac_browning": float((d < -0.05).sum() / n_valid),
                "n_valid_pixels": n_valid,
            }

        except Exception as exc:
            return {"error": str(exc)}

    # ── Convenience: best available ortho for bbox ─────────────────────────────

    def best_available(self) -> Optional[str]:
        """Return the newest collection that has tiles for this bbox."""
        for coll in COLLECTIONS_BY_YEAR:
            items = self.list_tiles(coll, limit=1)
            if items:
                logger.info(f"Best available ortho collection: {coll}")
                return coll
        return None
