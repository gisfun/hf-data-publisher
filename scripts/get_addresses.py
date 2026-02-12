import asyncio
import aiohttp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from huggingface_hub import HfApi
import os
import sys
import argparse

# CONFIGURATION
CONCURRENT_REQUESTS = 25  # Optimal throttle for OneMap stability
TOKEN = os.getenv("ONEMAP_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "gisfun/spatial-datasets"

async def fetch_pcode(pcode, session, semaphore):
    async with semaphore:
        url = f"https://www.onemap.gov.sg{pcode}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
        headers = {} # {"Authorization": TOKEN}
        
        for attempt in range(3): # Simple retry logic
            try:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    elif response.status == 429:
                        await asyncio.sleep(2 ** attempt) # Exponential backoff
                return []
            except Exception:
                await asyncio.sleep(1)
        return []

async def process_range(start, end):
    pcodes = [f"{p:06d}" for p in range(start, end + 1)]
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_pcode(p, session, semaphore) for p in pcodes]
        # Using a progress bar in logs
        results = await asyncio.gather(*tasks)
    
    flattened = [b for sublist in results for b in sublist]
    if not flattened: return None
    
    df = pd.DataFrame(flattened)
    # Ensure numeric types for DuckDB range queries
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    #df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    # Create GeoParquet with redundant Lat/Lon
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
        crs="EPSG:4326"
    )
    return gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int, help="Start of postal code range")
    parser.add_argument("end", type=int, help="End of postal code range")
    args = parser.parse_args()

    print(f"🚀 Processing: {args.start:06d} to {args.end:06d}")
    
    gdf = asyncio.run(process_range(args.start, args.end))
    
    if gdf is not None and not gdf.empty:
        fname = f"addresses_{args.start:06d}_{args.end:06d}.parquet"
        gdf.to_parquet(fname, index=False)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=f"chunks/{fname}",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"✅ Uploaded {fname} with {len(gdf)} records.")
    else:
        print("⚠️ No data found in this range.")
