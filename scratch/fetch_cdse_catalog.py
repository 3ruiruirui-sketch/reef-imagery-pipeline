#!/usr/bin/env python3
"""
fetch_cdse_catalog.py
========================================================================
Queries the official Copernicus Data Space Ecosystem (CDSE) catalog API.
This REST API replaces the old ESA SciHub/SciServer catalog API.
It searches for Sentinel-2 L2A datasets intersecting the Albufeira coastal
shelf (Pedra de Santa Eulália) with low cloud cover.
"""

import json
import requests
import pandas as pd

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

def search_algarve_sentinel2(year: int, max_cloud: float = 5.0):
    """
    Queries the CDSE OData catalog API for Sentinel-2 MSI L2A products.
    Points to Albufeira coordinates: Lat 37.068978, Lon -8.210328.
    """
    # Coordinate filter: Point intersects Albufeira coastal shelf (with SRID=4326)
    point_wkt = "SRID=4326;POINT(-8.210328 37.068978)"
    
    # Query parameters using standard OData filter syntax
    query_filter = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value lt {max_cloud}) and "
        f"ContentDate/Start ge {year}-07-01T00:00:00.000Z and "
        f"ContentDate/Start le {year}-09-30T23:59:59.000Z and "
        f"OData.CSC.Intersects(area=geography'{point_wkt}')"
    )
    
    params = {
        "$filter": query_filter,
        "$top": 20,
        "$orderby": "ContentDate/Start desc",
        "$expand": "Attributes"
    }
    
    print(f"📡 Querying CDSE catalog for Sentinel-2 images in {year} (Cloud Cover < {max_cloud}%)...")
    
    try:
        response = requests.get(CDSE_ODATA_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        products = data.get("value", [])
        print(f"✅ Found {len(products)} matching scenes.")
        
        records = []
        for p in products:
            scene_name = p.get("Name")
            start_date = p.get("ContentDate", {}).get("Start")
            prod_id = p.get("Id")
            
            # Extract cloud cover from nested attributes list
            cloud_cover = None
            for attr in p.get("Attributes", []):
                if attr.get("Name") == "cloudCover":
                    cloud_cover = attr.get("Value")
                    break
                    
            records.append({
                "Product_ID": prod_id,
                "Scene_Name": scene_name,
                "Acquisition_Date": start_date,
                "Cloud_Cover_Pct": cloud_cover
            })
            
        df = pd.DataFrame(records)
        print("\n", df.to_string(index=False))
        return df
        
    except requests.RequestException as e:
        print(f"❌ CDSE API query failed: {e}")
        return None

if __name__ == "__main__":
    search_algarve_sentinel2(year=2025)
