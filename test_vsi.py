import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

# Target coord
lat, lon = 37.05811, -8.20978

catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
search = catalog.search(
    collections=['sentinel-2-l2a'],
    intersects={'type': 'Point', 'coordinates': [lon, lat]},
    datetime="2024-09-30/2024-09-30"
)
item = list(search.items())[0]

# Get the B02 URL
b02_href = item.assets["B02"].href

with rasterio.Env(AWS_NO_SIGN_REQUEST='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
    with rasterio.open(b02_href) as src:
        # Convert lat/lon to raster CRS
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        
        # 1000m buffer
        window = from_bounds(x - 500, y - 500, x + 500, y + 500, src.transform)
        
        data = src.read(1, window=window)
        print("Read success! Shape:", data.shape)
        print("Mean B02:", data.mean())
