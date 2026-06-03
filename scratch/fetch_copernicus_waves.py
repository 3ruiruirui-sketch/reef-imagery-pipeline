#!/usr/bin/env python3
"""
fetch_copernicus_waves.py
========================================================================
Downloads regional waves data from the Copernicus Marine Service (CMEMS).
Uses the official 'copernicusmarine' Python toolbox client to extract:
  - VHM0 (Significant Wave Height, meters)
  - VTPK (Peak Wave Period, seconds)
  - VMDR (Mean Wave Direction, degrees)

Saves the extracted parameters to a local NetCDF and a tabular CSV file.
Area: Algarve Coast Bounding Box (Sagres to Vila Real de Santo António).
"""

import os
import copernicusmarine
import pandas as pd
import xarray as xr

# Define Algarve Region Bounding Box
# West: -9.0 (Sagres) | East: -7.4 (Vila Real de Santo António) | South: 36.8 | North: 37.1
ALGARVE_BBOX = {
    "min_lon": -9.0,
    "max_lon": -7.4,
    "min_lat": 36.8,
    "max_lat": 37.1
}

# Dataset: Iberian Biscay Irish (IBI) Ocean Wave Analysis and Forecast
# Temporal: hourly forecasts | Spatial: 0.05 degree resolution (~5.5 km)
DATASET_ID = "cmems_mod_ibi_wav_anfc_0.05deg-3hr_PT3H-i"
VARIABLES = ["VHM0", "VTPK", "VMDR"]  # Significant Wave Height, Peak Wave Period, Mean Wave Direction

def download_algarve_waves(start_date: str, end_date: str, output_dir: str = "outputs"):
    """
    Downloads regional waves data from Copernicus Marine Service (CMEMS).
    Saves the extracted parameters to a local NetCDF and CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    nc_filename = f"{output_dir}/algarve_waves_{start_date.replace('-', '')}.nc"
    csv_filename = f"{output_dir}/algarve_waves_{start_date.replace('-', '')}.csv"
    
    print(f"📡 Querying Copernicus Marine Service for dataset {DATASET_ID}...")
    
    try:
        # Download subset as NetCDF using the official CMEMS Toolbox Python API
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=VARIABLES,
            minimum_longitude=ALGARVE_BBOX["min_lon"],
            maximum_longitude=ALGARVE_BBOX["max_lon"],
            minimum_latitude=ALGARVE_BBOX["min_lat"],
            maximum_latitude=ALGARVE_BBOX["max_lat"],
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=output_dir,
            output_filename=os.path.basename(nc_filename),
            force_download=True
        )
        print(f"✅ Saved NetCDF output to: {nc_filename}")
        
        # Load the downloaded NetCDF file and convert to tabular CSV format
        ds = xr.open_dataset(nc_filename)
        df = ds.to_dataframe().reset_index()
        
        # Drop rows with NaN values (land pixels)
        df_clean = df.dropna(subset=VARIABLES)
        df_clean.to_csv(csv_filename, index=False)
        print(f"📊 Extracted tabular CSV containing {len(df_clean)} records to: {csv_filename}")
        
        # Summary statistics
        print("\n🌊 Algarve Coastal Waves Summary:")
        print(f"  • Avg Wave Height (VHM0): {df_clean['VHM0'].mean():.2f} meters")
        print(f"  • Max Wave Height (VHM0): {df_clean['VHM0'].max():.2f} meters")
        print(f"  • Avg Peak Wave Period (VTPK): {df_clean['VTPK'].mean():.1f} seconds")
        
    except Exception as e:
        print(f"❌ Failed to fetch wave data: {e}")
        print("\n💡 NOTE: You must have a registered account on Copernicus Marine Service.")
        print("   If you haven't, register here: https://data.marine.copernicus.eu/register")
        print("   Once registered, run 'copernicusmarine login' in your shell to authenticate.")

if __name__ == "__main__":
    # Test for Albufeira / Pedra do Alto target period
    download_algarve_waves("2025-09-25", "2025-09-26")
