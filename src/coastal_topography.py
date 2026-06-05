"""
Coastal topography feature extraction from DGT MDT-50cm LiDAR DTM.

This module integrates with the reef imagery pipeline to extract
coastal terrain features (slope, aspect) around dive sites that
correlate with sediment resuspension, swell exposure, and water visibility.

Usage:
    from coastal_topography import CoastalTopographyAnalyzer
    
    analyzer = CoastalTopographyAnalyzer(
        bbox=(-8.25, 37.04, -8.17, 37.10),
        output_dir="./outputs/coastal_features"
    )
    
    # Extract features for specific dive sites
    features_df = analyzer.extract_features_for_sites(
        sites=[
            ("pedra_sta_eulalia", 37.069081, -8.210242),
            ("albufeira_reef", 37.0690, -8.2105),
        ],
        buffer_m=1000
    )
    
    # Save to CSV / GeoJSON
    features_df.to_csv("coastal_features.csv", index=False)
    
References:
    - DGT MDT-50cm via STAC: https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items
    - Sediment resuspension linked to coastal topography: Mahadevan & Archer (2000)
    - Stumpf et al. (2003) on optical properties of shallow water
"""

import base64
import logging
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import requests
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.mask import mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from rasterstats import zonal_stats
    HAS_RASTERSTATS = True
except ImportError:
    HAS_RASTERSTATS = False

logger = logging.getLogger(__name__)


class CoastalTopographyAnalyzer:
    """Extract terrain features from DGT MDT-50cm or Copernicus GLO-30 around dive sites."""

    # DGT STAC endpoint
    STAC_URL = "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items"

    # Copernicus GLO-30 — public AWS bucket (no auth required, ~30 m resolution)
    GLO30_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

    # CDSE OAuth + download (uses user's Copernicus/CMEMS credentials)
    CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"

    # Credentials file written by copernicus-marine-client (base64-encoded)
    CDSE_CRED_FILE = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"

    # Native CRS of DGT MDT-50cm: ETRS89 / Portugal TM06
    NATIVE_CRS = "EPSG:3763"
    WGS84_CRS = "EPSG:4326"

    # Nodata value used by DGT
    NODATA_VALUE = -999.0

    def __init__(self,
                 bbox: Tuple[float, float, float, float],
                 output_dir: str = "./outputs/coastal_features",
                 cache_tiles: bool = True,
                 dem_source: str = "auto"):
        """
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 (lon, lat)
            output_dir: where to save tiles, mosaics, and feature tables
            cache_tiles: if True, reuse downloaded tiles across analyses
            dem_source: one of "dgt" (50cm, requires DGT S3 credentials),
                        "copernicus" (GLO-30 via CDSE, uses ~/.copernicusmarine creds),
                        "srtm" (GLO-30 public AWS, no auth),
                        "auto" (tries dgt → copernicus → srtm)
        """
        self.bbox = bbox
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_tiles = cache_tiles
        self.dem_source = dem_source

        self.tiles_dir = self.output_dir / "mdt_tiles"
        self.tiles_dir.mkdir(exist_ok=True)

        self.dem_mosaic_path = self.output_dir / "dem_mosaic_50cm.tif"
        self.slope_path = self.output_dir / "slope_50cm.tif"
        self.aspect_path = self.output_dir / "aspect_50cm.tif"

        logger.info("CoastalTopographyAnalyzer initialized")
        logger.info(f"  BBox (WGS84): {self.bbox}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  DEM source: {self.dem_source}")
    
    def fetch_stac_items(self, limit: int = 50) -> List[Dict]:
        """
        Query DGT STAC endpoint for MDT-50cm items in the bbox.
        
        Returns:
            List of STAC features
        """
        params = {
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "limit": limit,
            "f": "json",
        }
        
        logger.info(f"Querying DGT STAC: {self.STAC_URL}")
        try:
            r = requests.get(self.STAC_URL, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"STAC query failed: {e}")
            return []
        
        fc = r.json()
        n_returned = fc.get("context", {}).get("returned", 0)
        logger.info(f"DGT STAC returned {n_returned} features")
        
        return fc.get("features", [])
    
    def download_mdt_tiles(self, limit: int = 50) -> List[Path]:
        """
        Download MDT-50cm GeoTIFF tiles from DGT STAC.
        
        Args:
            limit: max number of items to request from STAC
            
        Returns:
            List of local file paths
        """
        features = self.fetch_stac_items(limit=limit)
        if not features:
            logger.warning("No STAC features found; no tiles to download")
            return []
        
        tile_paths = []
        for feat in features:
            item_id = feat.get("id", "unknown")
            asset = feat.get("assets", {}).get("Data", {})
            href = asset.get("href")
            
            if not href:
                logger.warning(f"Feature {item_id} has no Data.href")
                continue
            
            fname = Path(href).name
            local_path = self.tiles_dir / fname
            
            # Check cache
            if local_path.exists():
                logger.debug(f"Tile {item_id} already cached: {local_path}")
                tile_paths.append(local_path)
                continue
            
            # Download
            logger.info(f"Downloading {item_id} -> {fname}")
            try:
                with requests.get(href, stream=True, timeout=60) as r_tif:
                    r_tif.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r_tif.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                logger.info(f"Downloaded {fname} ({local_path.stat().st_size / 1e6:.1f} MB)")
                tile_paths.append(local_path)
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")
        
        return tile_paths

    # ── Copernicus GLO-30 helpers ──────────────────────────────────────────────

    @staticmethod
    def _read_cdse_credentials() -> Tuple[str, str]:
        """Read username/password from ~/.copernicusmarine/.copernicusmarine-credentials."""
        cred_file = CoastalTopographyAnalyzer.CDSE_CRED_FILE
        if not cred_file.exists():
            raise FileNotFoundError(f"CDSE credentials not found: {cred_file}")
        raw = base64.b64decode(cred_file.read_text().strip().rstrip("%")).decode()
        creds: Dict[str, str] = {}
        for line in raw.splitlines():
            if "=" in line and not line.startswith("["):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
        return creds["username"], creds["password"]

    def _get_cdse_token(self) -> str:
        """Obtain a short-lived OAuth2 bearer token from CDSE."""
        user, pwd = self._read_cdse_credentials()
        resp = requests.post(
            self.CDSE_TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "username": user,
                "password": pwd,
                "grant_type": "password",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    @staticmethod
    def _glo30_tiles_for_bbox(bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """Return list of (lat_floor, lon_floor) pairs covering bbox for 1°×1° GLO-30 tiles."""
        minx, miny, maxx, maxy = bbox
        tiles = []
        for lat in range(int(np.floor(miny)), int(np.floor(maxy)) + 1):
            for lon in range(int(np.floor(minx)), int(np.floor(maxx)) + 1):
                tiles.append((lat, lon))
        return tiles

    @staticmethod
    def _glo30_tile_name(lat: int, lon: int) -> str:
        """Return the Copernicus GLO-30 tile stem for a given (lat_floor, lon_floor)."""
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return (
            f"Copernicus_DSM_COG_10_{lat_dir}{abs(lat):02d}_00"
            f"_{lon_dir}{abs(lon):03d}_00_DEM"
        )

    def download_glo30_public(self) -> List[Path]:
        """
        Download Copernicus GLO-30 tiles from the public AWS S3 bucket (no auth).

        Resolution: ~30 m.  Covers the world.  Tiles are 1°×1°.
        """
        if not HAS_RASTERIO:
            logger.error("rasterio required for GLO-30 download")
            return []

        tiles = self._glo30_tiles_for_bbox(self.bbox)
        logger.info(f"GLO-30 public: {len(tiles)} tile(s) needed for bbox")

        paths = []
        for lat, lon in tiles:
            stem = self._glo30_tile_name(lat, lon)
            url = f"{self.GLO30_BASE_URL}/{stem}/{stem}.tif"
            local = self.tiles_dir / f"{stem}.tif"

            if local.exists() and self.cache_tiles:
                logger.info(f"  Cached: {local.name}")
                paths.append(local)
                continue

            logger.info(f"  Downloading GLO-30 tile: {stem}")
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(local, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk)
                logger.info(f"  Saved {local.name} ({local.stat().st_size / 1e6:.1f} MB)")
                paths.append(local)
            except Exception as exc:
                logger.error(f"  Failed to download {url}: {exc}")

        return paths

    def download_copernicus_cdse(self) -> List[Path]:
        """
        Download Copernicus DEM GLO-30 tiles via CDSE (authenticated).

        Uses ~/.copernicusmarine/.copernicusmarine-credentials for OAuth2.
        Falls back to public AWS bucket if a tile is not found on CDSE.
        """
        try:
            token = self._get_cdse_token()
            headers = {"Authorization": f"Bearer {token}"}
        except Exception as exc:
            logger.warning(f"CDSE auth failed ({exc}); falling back to public GLO-30")
            return self.download_glo30_public()

        tiles = self._glo30_tiles_for_bbox(self.bbox)
        logger.info(f"GLO-30 via CDSE: {len(tiles)} tile(s) needed")

        paths = []
        for lat, lon in tiles:
            stem = self._glo30_tile_name(lat, lon)
            local = self.tiles_dir / f"{stem}.tif"

            if local.exists() and self.cache_tiles:
                logger.info(f"  Cached: {local.name}")
                paths.append(local)
                continue

            # Search CDSE OData for this tile
            product_name = f"{stem}__30"
            search_url = (
                f"{self.CDSE_CATALOGUE_URL}/Products"
                f"?$filter=Collection/Name eq 'COP-DEM'"
                f" and contains(Name,'{stem}')"
                f"&$top=1"
            )
            try:
                sr = requests.get(search_url, headers=headers, timeout=30)
                sr.raise_for_status()
                products = sr.json().get("value", [])
            except Exception as exc:
                logger.warning(f"  CDSE search failed for {stem}: {exc}")
                products = []

            if products:
                pid = products[0]["Id"]
                dl_url = f"{self.CDSE_DOWNLOAD_URL}/Products({pid})/$value"
                logger.info(f"  Downloading {stem} from CDSE (id={pid})")
                try:
                    with requests.get(dl_url, headers=headers, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with open(local, "wb") as fh:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                if chunk:
                                    fh.write(chunk)
                    logger.info(f"  Saved {local.name} ({local.stat().st_size / 1e6:.1f} MB)")
                    paths.append(local)
                    continue
                except Exception as exc:
                    logger.warning(f"  CDSE download failed for {stem}: {exc}; trying public bucket")

            # Fallback to public AWS for this tile
            fallback = self.download_glo30_public()
            paths.extend(p for p in fallback if p not in paths)

        return paths

    def build_dem_mosaic(self, tile_paths: Optional[List[Path]] = None) -> Optional[Path]:
        """
        Merge MDT-50cm tiles into a single DEM GeoTIFF.
        
        If mosaic already exists and cache_tiles=True, skips rebuild.
        
        Args:
            tile_paths: list of local tile paths. If None, calls download_mdt_tiles()
            
        Returns:
            Path to mosaic GeoTIFF, or None on failure
        """
        if not HAS_RASTERIO:
            logger.error("rasterio not installed; cannot build mosaic")
            return None
        
        # Check cache
        if self.dem_mosaic_path.exists() and self.cache_tiles:
            logger.info(f"DEM mosaic already cached: {self.dem_mosaic_path}")
            return self.dem_mosaic_path
        
        # Download tiles if not provided — dispatch based on dem_source
        if tile_paths is None:
            source = self.dem_source
            if source == "dgt":
                tile_paths = self.download_mdt_tiles()
            elif source == "copernicus":
                tile_paths = self.download_copernicus_cdse()
            elif source == "srtm":
                tile_paths = self.download_glo30_public()
            else:  # "auto": DGT first, then CDSE, then public GLO-30
                tile_paths = self.download_mdt_tiles()
                if not tile_paths:
                    logger.info("DGT tiles unavailable; trying Copernicus CDSE...")
                    tile_paths = self.download_copernicus_cdse()
                if not tile_paths:
                    logger.info("CDSE unavailable; falling back to public GLO-30...")
                    tile_paths = self.download_glo30_public()

        if not tile_paths:
            logger.error("No tiles available for mosaicing")
            return None
        
        logger.info(f"Building mosaic from {len(tile_paths)} tiles...")

        try:
            import rioxarray as rxr
            from rasterio.crs import CRS as RioCRS

            src_files = [rasterio.open(str(p)) for p in tile_paths]

            # Merge tiles
            mosaic, mosaic_transform = merge(src_files)
            src_crs = src_files[0].crs
            meta = src_files[0].meta.copy()
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": mosaic_transform,
                "dtype": "float32",
                "nodata": self.NODATA_VALUE,
            })
            for src in src_files:
                src.close()

            # Write merged mosaic to a temp file, then reproject to EPSG:3763 if needed
            tmp_path = self.tiles_dir / "_mosaic_raw.tif"
            with rasterio.open(str(tmp_path), "w", **meta) as dst:
                dst.write(mosaic.astype("float32"))

            native_crs = RioCRS.from_epsg(3763)
            if src_crs != native_crs:
                logger.info(f"  Reprojecting mosaic from {src_crs} → EPSG:3763 ...")
                da = rxr.open_rasterio(str(tmp_path), masked=True)
                da_reproj = da.rio.reproject(self.NATIVE_CRS)
                da_reproj.rio.to_raster(str(self.dem_mosaic_path), dtype="float32")
                tmp_path.unlink(missing_ok=True)
                logger.info(f"  Reprojection complete")
            else:
                tmp_path.rename(self.dem_mosaic_path)

            with rasterio.open(str(self.dem_mosaic_path)) as chk:
                logger.info(f"Mosaic saved: {self.dem_mosaic_path}")
                logger.info(f"  Shape: {chk.shape}, CRS: {chk.crs}, res: {chk.res[0]:.1f} m")

            return self.dem_mosaic_path

        except Exception as e:
            logger.error(f"Mosaic creation failed: {e}")
            return None
    
    def derive_slope_aspect(self, 
                           dem_path: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Compute slope and aspect from DEM using gradient method.
        
        Slope is in degrees; aspect is in degrees (0=N, 90=E, 180=S, 270=W).
        
        Args:
            dem_path: path to DEM GeoTIFF. If None, uses self.dem_mosaic_path
            
        Returns:
            (slope_path, aspect_path) or (None, None) on failure
        """
        if not HAS_RASTERIO:
            logger.error("rasterio not installed")
            return None, None
        
        if dem_path is None:
            dem_path = self.dem_mosaic_path
        
        if not dem_path.exists():
            logger.error(f"DEM not found: {dem_path}")
            return None, None
        
        # Check cache
        if self.slope_path.exists() and self.aspect_path.exists() and self.cache_tiles:
            logger.info("Slope/aspect already cached")
            return self.slope_path, self.aspect_path
        
        logger.info(f"Computing slope and aspect from {dem_path}...")
        
        try:
            with rasterio.open(str(dem_path)) as src:
                dem = src.read(1, masked=True)
                transform = src.transform
                profile = src.profile
                
                # Resolution in meters
                res_x = abs(transform.a)
                res_y = abs(transform.e)
        
                logger.info(f"  DEM resolution: {res_x:.1f} m x {res_y:.1f} m")
            
            # Compute gradients (dz/dx, dz/dy in meters per meter)
            # np.gradient returns dy, dx
            dz_dy, dz_dx = np.gradient(dem.filled(np.nan), res_y, res_x)
            
            # Slope in degrees
            slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
            slope = np.degrees(slope_rad)
            
            # Aspect in degrees (0=N, 90=E, 180=S, 270=W)
            # Note: arctan2(x, -y) converts to geographic convention
            aspect_rad = np.arctan2(dz_dx, -dz_dy)
            aspect = np.degrees(aspect_rad)
            aspect = np.where(aspect < 0, 360 + aspect, aspect)
            
            # Save slope
            profile_out = profile.copy()
            profile_out.update(dtype="float32", nodata=np.nan)
            
            with rasterio.open(str(self.slope_path), "w", **profile_out) as dst:
                dst.write(slope.astype("float32"), 1)
            logger.info(f"Slope saved: {self.slope_path}")
            
            # Save aspect
            with rasterio.open(str(self.aspect_path), "w", **profile_out) as dst:
                dst.write(aspect.astype("float32"), 1)
            logger.info(f"Aspect saved: {self.aspect_path}")
            
            return self.slope_path, self.aspect_path
        
        except Exception as e:
            logger.error(f"Slope/aspect derivation failed: {e}")
            return None, None
    
    def extract_features_for_sites(self,
                                   sites: List[Tuple[str, float, float]],
                                   buffer_m: float = 1000,
                                   stats: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Extract terrain features (slope, aspect) for dive sites.
        
        Args:
            sites: list of (site_name, lat, lon) tuples
            buffer_m: circular buffer radius around each site (meters)
            stats: zonal statistics to compute. Default: ["mean", "median", "std", "min", "max", "percentile_90"]
            
        Returns:
            GeoDataFrame with features, or None on failure
        """
        if not HAS_RASTERSTATS:
            logger.error("rasterstats not installed; cannot compute zonal stats")
            logger.info("Install with: pip install rasterstats")
            return None
        
        if stats is None:
            stats = ["mean", "median", "std", "min", "max", "percentile_90"]
        
        # Build DEM and derive slope/aspect if needed
        if not self.dem_mosaic_path.exists():
            mosaic = self.build_dem_mosaic()
            if mosaic is None:
                return None
        
        if not self.slope_path.exists() or not self.aspect_path.exists():
            slope_path, aspect_path = self.derive_slope_aspect()
            if slope_path is None or aspect_path is None:
                return None
        
        logger.info(f"Extracting features for {len(sites)} sites with buffer={buffer_m}m...")
        
        # Convert sites to GeoDataFrame
        points = [Point(lon, lat) for _, lat, lon in sites]
        site_names = [name for name, _, _ in sites]
        
        gdf = gpd.GeoDataFrame(
            {"site_name": site_names},
            geometry=points,
            crs=self.WGS84_CRS
        )
        
        # Reproject to native CRS
        gdf_3763 = gdf.to_crs(self.NATIVE_CRS)
        gdf_3763["buffer_geometry"] = gdf_3763.geometry.buffer(buffer_m)
        
        # Extract zonal statistics
        features_list = []
        
        try:
            for idx, row in gdf_3763.iterrows():
                site_name = row["site_name"]
                buffer_geom = row["buffer_geometry"]
                
                logger.debug(f"Processing {site_name}...")
                
                # Slope stats
                slope_stats = zonal_stats(
                    [buffer_geom],
                    str(self.slope_path),
                    stats=stats,
                    nodata=self.NODATA_VALUE
                )[0]
                
                # Aspect stats (need circular mean)
                aspect_stats = zonal_stats(
                    [buffer_geom],
                    str(self.aspect_path),
                    stats=["mean", "median", "std", "min", "max"],
                    nodata=self.NODATA_VALUE
                )[0]
                
                # Build row
                feature_row = {
                    "site_name": site_name,
                    "latitude": gdf.loc[idx, "geometry"].y,
                    "longitude": gdf.loc[idx, "geometry"].x,
                    "buffer_m": buffer_m,
                }
                
                # Add slope stats with prefix
                for key, val in slope_stats.items():
                    feature_row[f"slope_{key}"] = val
                
                # Add aspect stats with prefix
                for key, val in aspect_stats.items():
                    feature_row[f"aspect_{key}"] = val
                
                features_list.append(feature_row)
        
        except Exception as e:
            logger.error(f"Zonal stats extraction failed: {e}")
            return None
        
        # Convert to DataFrame
        result_df = pd.DataFrame(features_list)
        result_df["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"Extracted features for {len(result_df)} sites")
        
        return result_df
    
    def save_features(self, 
                     features_df: pd.DataFrame,
                     output_name: str = "coastal_features") -> Dict[str, str]:
        """
        Save features to CSV and GeoJSON.
        
        Args:
            features_df: DataFrame with site features
            output_name: base name for output files
            
        Returns:
            dict with paths to saved files
        """
        csv_path = self.output_dir / f"{output_name}.csv"
        json_path = self.output_dir / f"{output_name}.json"
        geojson_path = self.output_dir / f"{output_name}.geojson"
        
        # CSV
        features_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # JSON (full dump)
        with open(json_path, "w") as f:
            json.dump(features_df.to_dict(orient="records"), f, indent=2, default=str)
        logger.info(f"Saved JSON: {json_path}")
        
        # GeoJSON (if geometry available)
        if "geometry" not in features_df.columns:
            gdf = gpd.GeoDataFrame(
                features_df,
                geometry=gpd.points_from_xy(features_df["longitude"], features_df["latitude"]),
                crs=self.WGS84_CRS
            )
        else:
            gdf = gpd.GeoDataFrame(features_df, crs=self.WGS84_CRS)
        
        gdf.to_file(geojson_path, driver="GeoJSON")
        logger.info(f"Saved GeoJSON: {geojson_path}")
        
        return {
            "csv": str(csv_path),
            "json": str(json_path),
            "geojson": str(geojson_path),
        }
    
    def run_analysis(self,
                    sites: List[Tuple[str, float, float]],
                    buffer_m: float = 1000,
                    output_name: str = "coastal_features") -> Dict:
        """
        Full pipeline: download tiles → mosaic → slope/aspect → features → save.
        
        Args:
            sites: list of (site_name, lat, lon) tuples
            buffer_m: buffer radius around each site (meters)
            output_name: base name for output files
            
        Returns:
            dict with status and file paths
        """
        logger.info("=" * 70)
        logger.info("COASTAL TOPOGRAPHY ANALYSIS PIPELINE")
        logger.info("=" * 70)
        
        try:
            # Step 1: Download tiles (source dispatched inside build_dem_mosaic)
            logger.info(f"\n[1/4] Acquiring DEM tiles (source={self.dem_source})...")
            tile_paths = None  # let build_dem_mosaic dispatch
            # Step 2: Mosaic (also triggers download if tile_paths is None)
            logger.info("\n[2/4] Building DEM mosaic...")
            mosaic_path = self.build_dem_mosaic(tile_paths)
            if mosaic_path is None:
                return {"status": "error", "message": "Mosaic creation failed"}
            
            # Step 3: Derive slope/aspect
            logger.info("\n[3/4] Deriving slope and aspect...")
            slope_path, aspect_path = self.derive_slope_aspect(mosaic_path)
            if slope_path is None or aspect_path is None:
                return {"status": "error", "message": "Slope/aspect derivation failed"}
            
            # Step 4: Extract features
            logger.info("\n[4/4] Extracting features for dive sites...")
            features_df = self.extract_features_for_sites(sites, buffer_m=buffer_m)
            if features_df is None:
                return {"status": "error", "message": "Feature extraction failed"}
            
            # Step 5: Save
            logger.info("\n[5/5] Saving results...")
            file_paths = self.save_features(features_df, output_name=output_name)
            
            logger.info("=" * 70)
            logger.info("ANALYSIS COMPLETE")
            logger.info("=" * 70)
            
            return {
                "status": "success",
                "sites_analyzed": len(features_df),
                "tiles_downloaded": len(tile_paths) if tile_paths else "n/a",
                "dem_mosaic": str(mosaic_path),
                "slope_raster": str(slope_path),
                "aspect_raster": str(aspect_path),
                "output_files": file_paths,
                "features_shape": features_df.shape,
                "features_columns": list(features_df.columns),
            }
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


def main():
    """Example usage with Algarve survey sites."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Survey sites from algarve_reef_survey.py
    survey_sites = [
        ("tavira_west",          37.1050, -7.6800),
        ("ilha_de_tavira",       37.0950, -7.7200),
        ("fuseta",               37.0600, -7.7600),
        ("olhao_offshore",       37.0350, -7.8200),
        ("faro_east",            37.0100, -7.8800),
        ("praia_de_faro",        36.9800, -7.9400),
        ("ancão_peninsula",      36.9650, -8.0000),
        ("quarteira",            37.0650, -8.1000),
        ("vilamoura",            37.0750, -8.1300),
        ("olhos_de_agua",        37.0900, -8.1600),
        ("pedra_sta_eulalia",    37.069081, -8.210242),
        ("albufeira_reef",       37.0690, -8.2105),
        ("galé",                 37.0560, -8.2296),
        ("salgados",             37.0950, -8.3000),
        ("armacao_de_pera",      37.0700, -8.3600),
    ]
    
    # BBox covering all sites (with margin)
    lats = [s[1] for s in survey_sites]
    lons = [s[2] for s in survey_sites]
    bbox = (min(lons) - 0.05, min(lats) - 0.05, max(lons) + 0.05, max(lats) + 0.05)
    
    analyzer = CoastalTopographyAnalyzer(
        bbox=bbox,
        output_dir="./outputs/coastal_topography"
    )
    
    result = analyzer.run_analysis(
        sites=survey_sites,
        buffer_m=4000,
        output_name="algarve_coastal_features"
    )
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
