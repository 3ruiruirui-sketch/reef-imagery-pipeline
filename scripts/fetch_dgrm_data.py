#!/usr/bin/env python3
"""
Fetch marine data from DGRM (Direção-Geral de Recursos Marítimos) ArcGIS server.
https://webgis.dgrm.mm.gov.pt/arcgis/rest/services

Useful for reef imagery pipeline:
- Coral patch locations
- Marine Protected Areas (OSPAR)
- Bathymetric contours
- EBSA (Ecologically Significant Areas)
"""

import json
import urllib.request
from pathlib import Path

# Base URL for DGRM ArcGIS server
DGRM_BASE = "https://webgis.dgrm.mm.gov.pt/arcgis/rest/services"

# Services of interest
SERVICES = {
    "amp_ampere_coral_patch": {
        "url": f"{DGRM_BASE}/GT_AMP/AMP_Ampere_Coral_Patch/MapServer/0",
        "description": "Ampere Seamount coral patch data",
        "type": "coral",
    },
    "amp_ospar": {
        "url": f"{DGRM_BASE}/GEOPORTAL_MAR_PORTUGUES/AMP_OSPAR/MapServer/0",
        "description": "OSPAR Marine Protected Areas",
        "type": "mpa",
    },
    "ebsa": {
        "url": f"{DGRM_BASE}/GEOPORTAL_MAR_PORTUGUES/EBSA_to_geoportal_MP/MapServer",
        "description": "Ecologically or Biologically Significant Marine Areas",
        "type": "ebsa",
    },
    "bathymetry_500_1000": {
        "url": f"{DGRM_BASE}/GEOPORTAL_MAR_PORTUGUES/Arrasto_batim_1000_500_servi%C3%A7o_mapa/MapServer",
        "description": "Bathymetric contours (500m and 1000m intervals)",
        "type": "bathymetry",
    },
}


def query_features(url):
    """Query ArcGIS Feature Service and return GeoJSON."""
    # Note: resultRecordCount causes pagination errors on some services
    query_url = f"{url}/query?where=1%3D1&outFields=*&f=geojson"
    try:
        with urllib.request.urlopen(query_url, timeout=60) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"  Error querying {url}: {e}")
        return None


def fetch_service_info(url):
    """Fetch service metadata."""
    info_url = f"{url}?f=json"
    try:
        with urllib.request.urlopen(info_url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"  Error fetching info from {url}: {e}")
        return None


def main():
    output_dir = Path("data/dgrm")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching marine data from DGRM ArcGIS server...\n")

    for key, service in SERVICES.items():
        print(f"--- {service['description']} ---")
        print(f"URL: {service['url']}")

        # Fetch layer info
        info = fetch_service_info(service['url'])
        if info:
            # Save metadata
            meta_file = output_dir / f"{key}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"  Saved metadata: {meta_file}")

            # Try to query features (if it's a feature layer, not a folder)
            if '/MapServer/' in service['url']:
                features = query_features(service['url'])
                if features:
                    geojson_file = output_dir / f"{key}_features.geojson"
                    with open(geojson_file, 'w') as f:
                        json.dump(features, f, indent=2)
                    print(f"  Saved features: {geojson_file}")
                    if 'features' in features:
                        print(f"  Found {len(features['features'])} features")
        print()

    print("Done! Data saved to data/dgrm/")


if __name__ == "__main__":
    main()
