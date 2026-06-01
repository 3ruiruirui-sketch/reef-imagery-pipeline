import requests
import xml.etree.ElementTree as ET

urls = [
    "https://ortos.dgterritorio.gov.pt/wms/ortosat2023?service=wms&request=getcapabilities",
    "https://ortos.dgterritorio.gov.pt/wms/orto2021?service=wms&request=getcapabilities",
    "https://ortos.dgterritorio.gov.pt/wms/orto2018?service=wms&request=getcapabilities"
]

for url in urls:
    print(f"\n--- Checking WMS: {url} ---")
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            # Find all layers
            layers = []
            for elem in root.iter():
                if elem.tag.endswith('Layer'):
                    name_elem = elem.find('{http://www.opengis.net/wms}Name')
                    title_elem = elem.find('{http://www.opengis.net/wms}Title')
                    if name_elem is not None and title_elem is not None:
                        layers.append((name_elem.text, title_elem.text))
            
            print(f"Status: Success, Found {len(layers)} layers:")
            for l in layers[:10]:
                print(f"  Name: {l[0]} | Title: {l[1]}")
        else:
            print(f"Status: HTTP {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")
