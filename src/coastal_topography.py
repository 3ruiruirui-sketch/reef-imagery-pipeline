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
    """Extract terrain features from DGT MDT-50cm around dive sites."""
    
    # DGT STAC endpoint
    STAC_URL = "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items"
    
    # Native CRS of DGT MDT-50cm: ETRS89 / Portugal TM06
    NATIVE_CRS = "EPSG:3763"
    WGS84_CRS = "EPSG:4326"
    
    # Nodata value used by DGT
    NODATA_VALUE = -999.0
    
    def __init__(self, 
                 bbox: Tuple[float, float, float, float],
                 output_dir: str = "./outputs/coastal_features",
                 cache_tiles: bool = True):
        """
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 (lon, lat)
            output_dir: where to save tiles, mosaics, and feature tables
            cache_tiles: if True, reuse downloaded tiles across analyses
        """
        self.bbox = bbox
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_tiles = cache_tiles
        
        self.tiles_dir = self.output_dir / "mdt_tiles"
        self.tiles_dir.mkdir(exist_ok=True)
        
        self.dem_mosaic_path = self.output_dir / "dem_mosaic_50cm.tif"
        self.slope_path = self.output_dir / "slope_50cm.tif"
        self.aspect_path = self.output_dir / "aspect_50cm.tif"
        
        logger.info(f"CoastalTopographyAnalyzer initialized")
        logger.info(f"  BBox (WGS84): {self.bbox}")
        logger.info(f"  Output: {self.output_dir}")
    
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
        
        # Download tiles if not provided
        if tile_paths is None:
            tile_paths = self.download_mdt_tiles()
        
        if not tile_paths:
            logger.error("No tiles available for mosaicing")
            return None
        
        logger.info(f"Building mosaic from {len(tile_paths)} tiles...")
        
        try:
            src_files = [rasterio.open(str(p)) for p in tile_paths]
            
            # Merge
            mosaic, mosaic_transform = merge(src_files)
            meta = src_files[0].meta.copy()
            meta.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": mosaic_transform,
                "dtype": mosaic.dtype,
                "nodata": self.NODATA_VALUE,
            })
            
            with rasterio.open(str(self.dem_mosaic_path), "w", **meta) as dst:
                dst.write(mosaic)
            
            for src in src_files:
                src.close()
            
            logger.info(f"Mosaic saved: {self.dem_mosaic_path}")
            logger.info(f"  Shape: {mosaic.shape}, CRS: {meta['crs']}")
            
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
            # Step 1: Download tiles
            logger.info("\n[1/4] Downloading MDT-50cm tiles from DGT STAC...")
            tile_paths = self.download_mdt_tiles()
            if not tile_paths:
                return {"status": "error", "message": "No tiles downloaded"}
            
            # Step 2: Mosaic
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
                "tiles_downloaded": len(tile_paths),
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
        buffer_m=1000,
        output_name="algarve_coastal_features"
    )
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
