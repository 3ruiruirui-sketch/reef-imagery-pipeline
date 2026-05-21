import argparse
import os
import urllib.request
import requests
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
from datetime import datetime

# --- Configuration ---
TARGET_LAT = 37.069071
TARGET_LON = -8.210492
OUTPUT_DIR = "reef_output"

# Best OP20 dates ranked by Secchi depth
OP20_DATES = [
    "2024-10-15",
    "2021-10-31",
    "2022-09-12",
    "2024-01-03",
    "2023-10-26"
]

# DGT WCS Base URL (Approximated - check dgt_capabilities_2018.xml for the exact service)
DGT_WCS_URL = "https://snig.dgterritorio.gov.pt/ows/ows" 

def setup_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def step_capabilities():
    print("-> Probing DGT WCS capabilities...")
    # This is a sample URL - in a real scenario you would query the specific 2018 orthophoto WCS
    url = f"{DGT_WCS_URL}?service=WCS&request=GetCapabilities"
    
    try:
        response = requests.get(url)
        output_file = os.path.join(OUTPUT_DIR, "dgt_capabilities_2018.xml")
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"   Saved to {output_file}")
    except Exception as e:
        print(f"   Error fetching capabilities: {e}")

def step_ortho():
    print("-> Downloading DGT 2018 25 cm orthophoto clip...")
    # Note: Proper WCS GetCoverage request requires knowing the exact CoverageId from GetCapabilities,
    # as well as bounding box in the correct CRS (e.g. EPSG:3763 for Portugal).
    # Here we simulate the process or provide a constructed URL.
    coverage_id = "Ortofotomapa_2018" # Example ID
    
    # Approx 500m bounding box (simple degree math for demo)
    deg_500m = 0.0045
    min_lon, min_lat = TARGET_LON - deg_500m, TARGET_LAT - deg_500m
    max_lon, max_lat = TARGET_LON + deg_500m, TARGET_LAT + deg_500m
    
    url = (f"{DGT_WCS_URL}?service=WCS&request=GetCoverage&version=2.0.1"
           f"&coverageId={coverage_id}&format=image/tiff"
           f"&subset=Long({min_lon},{max_lon})&subset=Lat({min_lat},{max_lat})")
    
    output_file = os.path.join(OUTPUT_DIR, "dgt_ortho_2018_reef.tif")
    try:
        print(f"   Requesting: {url}")
        # In a real run, this would download the TIFF. Using a placeholder for now.
        # response = requests.get(url)
        # with open(output_file, "wb") as f:
        #     f.write(response.content)
        print(f"   [Simulated] Saved to {output_file}")
    except Exception as e:
        print(f"   Error fetching orthophoto: {e}")

def step_sentinel():
    print("-> Querying Copernicus STAC for Sentinel-2 L2A...")
    # Using Planetary Computer STAC API as a reliable open STAC catalog for Sentinel-2
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    # We will search for the best date (first in OP20_DATES)
    best_date = OP20_DATES[0]
    time_range = f"{best_date}T00:00:00Z/{best_date}T23:59:59Z"
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [TARGET_LON, TARGET_LAT]},
        datetime=time_range
    )
    
    items = list(search.items())
    if not items:
        print(f"   No Sentinel-2 imagery found for {best_date}.")
        return
        
    item = items[0]
    print(f"   Found Sentinel-2 item: {item.id} from {best_date}")
    
    # In a real scenario, you'd download the jp2 files. Here we simulate it or fetch small thumbnails.
    b02_url = item.assets["B02"].href
    b03_url = item.assets["B03"].href
    
    date_str = best_date.replace("-", "")
    b02_file = os.path.join(OUTPUT_DIR, f"S2_B02_{date_str}.jp2")
    b03_file = os.path.join(OUTPUT_DIR, f"S2_B03_{date_str}.jp2")
    
    print(f"   [Simulated] Downloading B02 to {b02_file}")
    print(f"   [Simulated] Downloading B03 to {b03_file}")

def step_ratio():
    print("-> Computing log(B02)/log(B03) contrast ratio...")
    best_date = OP20_DATES[0].replace("-", "")
    b02_file = os.path.join(OUTPUT_DIR, f"S2_B02_{best_date}.jp2")
    b03_file = os.path.join(OUTPUT_DIR, f"S2_B03_{best_date}.jp2")
    
    # For demonstration, we create a dummy array if files don't exist
    print(f"   Processing B02 and B03 for {best_date}...")
    ratio_file = os.path.join(OUTPUT_DIR, f"ratio_B02_B03_{best_date}.tif")
    plot_file = os.path.join(OUTPUT_DIR, f"ratio_analysis_{best_date}.png")
    
    # Simulate processing and output
    print(f"   [Simulated] Computed ratio and saved to {ratio_file}")
    print(f"   [Simulated] Visualisation saved to {plot_file}")

def step_gee():
    print("-> Writing Google Earth Engine export script...")
    gee_script = f"""// GEE Export Script for Albufeira Reef
// Center: {TARGET_LAT}, {TARGET_LON}

var point = ee.Geometry.Point([{TARGET_LON}, {TARGET_LAT}]);
var buffer = point.buffer(500);

var dataset = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(buffer)
                  .filterDate('{OP20_DATES[0]}', '{OP20_DATES[0]}T23:59:59')
                  .first();

if (dataset) {{
  var clipped = dataset.clip(buffer);
  
  // Compute log(B02)/log(B03)
  var b2 = clipped.select('B2');
  var b3 = clipped.select('B3');
  var ratio = b2.log().divide(b3.log()).rename('ratio');
  
  Export.image.toDrive({{
    image: ratio,
    description: 'reef_ratio_export',
    scale: 10,
    region: buffer
  }});
  print('Export task created. Check Tasks tab.');
}} else {{
  print('No image found for this date.');
}}
"""
    output_file = os.path.join(OUTPUT_DIR, "gee_reef_export.js")
    with open(output_file, "w") as f:
        f.write(gee_script)
    print(f"   Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Reef Imagery Acquisition Pipeline")
    parser.add_argument("--step", type=str, required=True, 
                        choices=["capabilities", "ortho", "sentinel", "ratio", "gee", "all"],
                        help="Step to execute")
    args = parser.parse_args()

    setup_output_dir()

    if args.step in ["capabilities", "all"]:
        step_capabilities()
    if args.step in ["ortho", "all"]:
        step_ortho()
    if args.step in ["sentinel", "all"]:
        step_sentinel()
    if args.step in ["ratio", "all"]:
        step_ratio()
    if args.step in ["gee", "all"]:
        step_gee()

if __name__ == "__main__":
    main()
