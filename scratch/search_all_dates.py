import sys
import os
import pandas as pd
from datetime import datetime
import planetary_computer as pc
from pystac_client import Client

def search_dates_for_site(lat, lon, site_name, years=5, max_cloud=30):
    print(f"\n🔍 Searching dates for {site_name} (GPS: {lat:.6f}, {lon:.6f})...")
    catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
    end_date = datetime.now()
    start_date = datetime(end_date.year - years, 1, 1)

    search = catalog.search(
        collections=['sentinel-2-l2a'],
        intersects={'type': 'Point', 'coordinates': [lon, lat]},
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query={"eo:cloud_cover": {"lt": max_cloud}}
    )

    items = list(search.items())
    print(f"   Found {len(items)} scenes with STAC cloud cover < {max_cloud}%")
    
    data = []
    for item in items:
        props = item.properties
        if props.get('s2:nodata_pixel_percentage', 100) > 20: 
            continue
        data.append({
            'date': item.datetime.strftime('%Y-%m-%d'),
            'cloud_cover': props.get('eo:cloud_cover', 100),
            'sun_elevation': props.get('view:sun_elevation', 45)
        })
        
    df = pd.DataFrame(data)
    if df.empty:
        print("   No scenes passed the nodata check.")
        return []
    
    df = df.sort_values('date', ascending=False).drop_duplicates('date')
    print(f"   Unique dates available: {len(df)}")
    return df

# Pedra de Santa Eulália
eulalia_df = search_dates_for_site(37.068978, -8.210328, "Pedra de Santa Eulalia")

# Pedra do Alto
alto_df = search_dates_for_site(37.058150, -8.209820, "Pedra do Alto")

print("\n--- Summary of Santa Eulalia (top 20 dates sorted by date) ---")
print(eulalia_df.head(20).to_string(index=False))

print("\n--- Summary of Pedra do Alto (top 20 dates sorted by date) ---")
print(alto_df.head(20).to_string(index=False))
