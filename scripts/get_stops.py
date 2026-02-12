import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xml.etree.ElementTree as ET
import requests
from huggingface_hub import HfApi
import os

# 1. Download and Parse XML
url = "https://www.lta.gov.sg/map/busService/bus_stops.xml"
response = requests.get(url)
root = ET.fromstring(response.content)

data = []
for stop in root.findall('busstop'):
    # Standardize data types
    lat = float(stop.findtext('coordinates/lat'))
    lon = float(stop.findtext('coordinates/long'))
    
    data.append({
        "name": stop.get('name'),
        "wab": stop.get('wab') == "true",
        "details": stop.findtext('details'),
        "geometry": Point(lon, lat)  # Create Shapely Point (Lon, Lat order!)
    })

# 2. Convert to GeoDataFrame and save as GeoParquet
gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
gdf.to_parquet("bus_stops.parquet", index=False)

# 3. Push to Hugging Face
api = HfApi()
api.upload_file(
    path_or_fileobj="bus_stops.parquet",
    path_in_repo="bus_stops.parquet",
    repo_id="gisfun/spatial-datasets",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
