"""
DGT MDT-50cm + Sentinel-2 integrator for reef imagery analysis.

Consumes:
  - MDT-50cm (50 cm LiDAR DTM) from DGT STAC endpoint
  - Sentinel-2 MSI from Copernicus (via sentinelhub-py or rasterio)
  
Outputs:
  - Aligned GeoTIFF mosaic (MDT + Sentinel bands) in EPSG:3763
  - Metadata JSON with crs, bounds, resolution, source info
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import requests
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)


class DGTSentinelIntegrator:
    """Download and align MDT-50cm (DGT) + Sentinel-2 for reef imagery."""
    
    # DGT STAC endpoint
    DGT_STAC_URL = "https://dgt-be.a.incd.pt:8081/collections/MDT-50cm/items"
    
    # Target CRS: ETRS89 / Portugal TM06 (used by DGT MDT-50cm)
    TARGET_CRS = "EPSG:3763"
    
    def __init__(self, 
                 bbox: Tuple[float, float, float, float],
                 output_dir: str = "./data/dgt_sentinel_integrated",
                 mdt_nodata: float = -999.0):
        """
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84 (lon, lat)
            output_dir: where to save tiles and mosaics
            mdt_nodata: nodata value for MDT-50cm
        """
        self.bbox = bbox
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mdt_nodata = mdt_nodata
        
        self.mdt_dir = self.output_dir / "mdt50cm_tiles"
        self.mdt_dir.mkdir(exist_ok=True)
        
        logger.info(f"DGT-Sentinel Integrator initialized. Output: {self.output_dir}")
        logger.info(f"BBox (WGS84): {self.bbox}")
    
    def download_mdt50cm_tiles(self, limit: int = 50) -> List[str]:
        """
        Download MDT-50cm tiles from DGT STAC for the given bbox.
        
        Args:
            limit: max number of items to request
            
        Returns:
            List of local file paths (GeoTIFFs)
        """
        params = {
            "bbox": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",
            "limit": limit,
            "f": "json",
        }
        
        logger.info(f"Querying DGT STAC: {self.DGT_STAC_URL}")
        r = requests.get(self.DGT_STAC_URL, params=params, timeout=30)
        r.raise_for_status()
        
        fc = r.json()
        n_returned = fc.get("context", {}).get("returned", 0)
        logger.info(f"DGT STAC returned {n_returned} features")
        
        if n_returned == 0:
            logger.warning("No MDT-50cm tiles found for this bbox!")
            return []
        
        tif_paths = []
        for feat in fc.get("features", []):
            item_id = feat.get("id", "unknown")
            asset = feat.get("assets", {}).get("Data", {})
            href = asset.get("href")
            
            if not href:
                logger.warning(f"Feature {item_id} has no Data.href, skipping")
                continue
            
            fname = os.path.basename(href)
            local_path = self.mdt_dir / fname
            
            if local_path.exists():
                logger.info(f"Tile {item_id} already exists: {local_path}")
                tif_paths.append(str(local_path))
                continue
            
            logger.info(f"Downloading {item_id} -> {local_path}")
            try:
                with requests.get(href, stream=True, timeout=60) as r_tif:
                    r_tif.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r_tif.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                logger.info(f"Successfully downloaded {fname}")
                tif_paths.append(str(local_path))
            except Exception as e:
                logger.error(f"Failed to download {href}: {e}")
        
        return tif_paths
    
    def mosaic_mdt50cm(self, tif_paths: List[str]) -> xr.DataArray:
        """
        Load and mosaic MDT-50cm tiles.
        
        Assumes all tiles are in EPSG:3763 with same resolution.
        
        Args:
            tif_paths: list of local GeoTIFF paths
            
        Returns:
            xarray DataArray with shape (y, x)
        """
        if not tif_paths:
            raise ValueError("No tiles provided for mosaicing")
        
        logger.info(f"Loading {len(tif_paths)} MDT-50cm tiles...")
        
        rasters = []
        for p in tif_paths:
            try:
                ds = rxr.open_rasterio(p, masked=True)
                # Squeeze out band dimension if it's 1
                if ds.sizes.get("band", 1) == 1:
                    ds = ds.squeeze("band")
                rasters.append(ds)
                logger.info(f"Loaded {p}, shape: {ds.shape}, CRS: {ds.rio.crs}")
            except Exception as e:
                logger.error(f"Failed to load {p}: {e}")
        
        if not rasters:
            raise ValueError("Could not load any raster files")
        
        # Ensure all in same CRS
        target_crs = rasters[0].rio.crs
        rasters_reproj = []
        for ds in rasters:
            if ds.rio.crs != target_crs:
                logger.info(f"Reprojecting to {target_crs}")
                ds = ds.rio.reproject(target_crs)
            rasters_reproj.append(ds)
        
        # Merge by finding common bounds
        logger.info("Merging MDT tiles...")
        combined = xr.concat(rasters_reproj, dim="tile")
        
        # Compute common bounds
        bounds_list = [ds.rio.bounds() for ds in rasters_reproj]
        minx = min(b[0] for b in bounds_list)
        miny = min(b[1] for b in bounds_list)
        maxx = max(b[2] for b in bounds_list)
        maxy = max(b[3] for b in bounds_list)
        
        # Clip to bbox bounds
        mosaic = combined.rio.clip_box(minx, miny, maxx, maxy)
        
        logger.info(f"Mosaiced MDT shape: {mosaic.shape}, bounds: ({minx}, {miny}, {maxx}, {maxy})")
        return mosaic
    
    def fetch_sentinel2_copernicus(self,
                                   date_start: str = "2024-01-01",
                                   date_end: str = "2024-12-31",
                                   cloud_pct: int = 30) -> Optional[xr.DataArray]:
        """
        Fetch Sentinel-2 L2A from Copernicus via sentinelhub-py or Copernicus Browser.
        
        For this example, we recommend using sentinelhub-py with a valid Copernicus account.
        
        Args:
            date_start: ISO date string
            date_end: ISO date string
            cloud_pct: max cloud coverage %
            
        Returns:
            xarray DataArray with Sentinel-2 bands (B02, B03, B04, B08, etc.) in EPSG:3763
        """
        # This is a template; sentinelhub-py setup requires credentials
        logger.info(
            f"Sentinel-2 fetch (template): bbox={self.bbox}, "
            f"date_range=[{date_start}, {date_end}], cloud_max={cloud_pct}%"
        )
        
        try:
            from sentinelhub import SentinelHubRequest, DataCollection, MimeType
            from sentinelhub import BBox, CRS
        except ImportError:
            logger.warning(
                "sentinelhub not installed. Install with: pip install sentinelhub\n"
                "Alternatively, download Sentinel-2 L2A from Copernicus Browser manually."
            )
            return None
        
        logger.info("Using sentinelhub-py to fetch Sentinel-2 L2A...")
        
        # BBox in WGS84
        bbox_wgs84 = BBox(self.bbox, CRS.WGS84)
        
        # Request all standard bands
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "B11", "B12"],
                output: {bands: 6}
            };
        }
        function evaluatePixel(sample) {
            return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
        }
        """
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    DataCollection.SENTINEL2_L2A,
                    from_date=date_start,
                    to_date=date_end,
                    mosaicking_order="mostRecent",
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_wgs84,
            size=(512, 512),  # Adjust resolution as needed
            config=None,  # uses ~/.sentinelhub/config.json
        )
        
        try:
            img = request.get_data()
            logger.info(f"Downloaded Sentinel-2 image: shape {img[0].shape}")
            # Convert to xarray
            # This requires additional metadata setup; for now return placeholder
            return None
        except Exception as e:
            logger.error(f"sentinelhub request failed: {e}")
            return None
    
    def reproject_match(self, 
                       src_da: xr.DataArray, 
                       reference_da: xr.DataArray) -> xr.DataArray:
        """
        Reproject src_da to match reference_da's CRS and grid.
        """
        if src_da.rio.crs != reference_da.rio.crs:
            logger.info(f"Reprojecting from {src_da.rio.crs} to {reference_da.rio.crs}")
            src_da = src_da.rio.reproject(reference_da.rio.crs)
        
        logger.info("Resampling to reference grid...")
        src_resampled = src_da.rio.reproject_match(reference_da)
        return src_resampled
    
    def stack_mdt_sentinel(self,
                          mdt: xr.DataArray,
                          sentinel: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Stack MDT-50cm + Sentinel-2 bands into a single DataArray.
        
        If sentinel is None, returns MDT only.
        """
        if sentinel is None:
            logger.warning("No Sentinel-2 data; returning MDT-50cm only")
            return mdt.rename("MDT_50cm_elevation")
        
        # Stack along 'band' dimension
        mdt_band = mdt.assign_coords(band=0).expand_dims("band")
        
        # Assume sentinel has multiple bands
        stacked = xr.concat([mdt_band, sentinel], dim="band")
        stacked = stacked.assign_coords(
            band=["MDT_50cm", "S2_B02", "S2_B03", "S2_B04", "S2_B08", "S2_B11", "S2_B12"][:stacked.sizes["band"]]
        )
        return stacked
    
    def save_mosaic_geotiff(self, 
                           data: xr.DataArray,
                           output_path: str,
                           metadata: Optional[Dict] = None) -> str:
        """
        Save xarray DataArray to GeoTIFF.
        """
        output_path = Path(output_path)
        logger.info(f"Saving mosaic to {output_path}")
        
        data.rio.to_raster(str(output_path))
        
        # Save companion JSON metadata
        if metadata:
            meta_path = output_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {meta_path}")
        
        return str(output_path)
    
    def integrate(self,
                  fetch_sentinel: bool = False,
                  date_start: str = "2024-01-01",
                  date_end: str = "2024-12-31",
                  cloud_pct: int = 30) -> Dict:
        """
        Full integration pipeline:
          1. Download MDT-50cm tiles from DGT STAC
          2. Mosaic MDT-50cm
          3. Optionally fetch Sentinel-2 and reproject to match
          4. Stack both and save output GeoTIFF
        
        Args:
            fetch_sentinel: whether to fetch Sentinel-2
            date_start, date_end: date range for Sentinel-2
            cloud_pct: max cloud coverage for Sentinel-2
            
        Returns:
            dict with paths and metadata
        """
        logger.info("=" * 60)
        logger.info("Starting DGT-Sentinel Integration Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Download MDT-50cm
        logger.info("\n[1/4] Downloading MDT-50cm tiles from DGT STAC...")
        tif_paths = self.download_mdt50cm_tiles()
        if not tif_paths:
            logger.error("No MDT tiles downloaded; aborting")
            return {"status": "error", "message": "No MDT tiles found"}
        
        # Step 2: Mosaic MDT-50cm
        logger.info("\n[2/4] Mosaicing MDT-50cm tiles...")
        mdt_mosaic = self.mosaic_mdt50cm(tif_paths)
        mdt_out = self.output_dir / "MDT_50cm_mosaic_algarve.tif"
        self.save_mosaic_geotiff(
            mdt_mosaic,
            str(mdt_out),
            {
                "source": "DGT MDT-50cm STAC",
                "crs": self.TARGET_CRS,
                "bbox_wgs84": self.bbox,
                "resolution_m": 0.5,
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        # Step 3: Fetch Sentinel-2 (optional)
        sentinel_mosaic = None
        if fetch_sentinel:
            logger.info("\n[3/4] Fetching Sentinel-2 L2A from Copernicus...")
            sentinel_mosaic = self.fetch_sentinel2_copernicus(date_start, date_end, cloud_pct)
            
            if sentinel_mosaic is not None:
                # Reproject to match MDT
                sentinel_mosaic = self.reproject_match(sentinel_mosaic, mdt_mosaic)
        else:
            logger.info("\n[3/4] Skipping Sentinel-2 fetch (fetch_sentinel=False)")
        
        # Step 4: Stack and save
        logger.info("\n[4/4] Stacking and saving integrated dataset...")
        stacked = self.stack_mdt_sentinel(mdt_mosaic, sentinel_mosaic)
        stacked_out = self.output_dir / "integrated_mdt_sentinel_algarve.tif"
        self.save_mosaic_geotiff(
            stacked,
            str(stacked_out),
            {
                "source": "DGT MDT-50cm + Sentinel-2 L2A",
                "crs": self.TARGET_CRS,
                "bbox_wgs84": self.bbox,
                "mdt_resolution_m": 0.5,
                "sentinel_resolution_m": 10,
                "bands": list(stacked.coords["band"].values),
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.info("=" * 60)
        logger.info("Integration complete!")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "mdt_mosaic_path": str(mdt_out),
            "integrated_path": str(stacked_out),
            "mdt_shape": mdt_mosaic.shape,
            "integrated_shape": stacked.shape,
            "crs": self.TARGET_CRS,
            "bbox_wgs84": self.bbox,
        }


def main():
    """Example usage."""
    import logging as lg
    lg.basicConfig(
        level=lg.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Algarve, Boliqueime/Faro area
    bbox = (-8.25, 37.04, -8.17, 37.10)
    
    integrator = DGTSentinelIntegrator(
        bbox=bbox,
        output_dir="./outputs/dgt_sentinel"
    )
    
    # Run full pipeline (fetch_sentinel=False for now, requires credentials)
    result = integrator.integrate(fetch_sentinel=False)
    
    print("\nIntegration Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
