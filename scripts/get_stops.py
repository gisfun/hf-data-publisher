import pandas as pd
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
    data.append({
        "name": stop.get('name'),
        "wab": stop.get('wab') == "true",
        "details": stop.findtext('details'),
        "lat": float(stop.findtext('coordinates/lat')),
        "long": float(stop.findtext('coordinates/long'))
    })

# 2. Convert to Parquet
df = pd.DataFrame(data)
df.to_parquet("bus_stops.parquet", index=False)

# 3. Push to Hugging Face
api = HfApi()
api.upload_file(
    path_or_fileobj="bus_stops.parquet",
    path_in_repo="bus_stops.parquet",
    repo_id="gisfun/spatial-data",  # <-- UPDATE THIS
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)
