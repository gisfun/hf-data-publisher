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
CONCURRENT_REQUESTS = 25  # Number of parallel request slots
TOKEN = os.getenv("ONEMAP_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "gisfun/spatial-datasets"

async def fetch_pcode(pcode, session, semaphore):
    async with semaphore:
        page = 1
        all_results = []
        headers = {} # {"Authorization": TOKEN}

        while True:
            # THE THROTTLE: Applied per page fetch to be safe
            await asyncio.sleep(1.0) 
            
            url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={pcode}&returnGeom=Y&getAddrDetails=Y&pageNum={page}"
            
            for attempt in range(4):
                try:
                    async with session.get(url, headers=headers, timeout=15) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            results = data.get("results", [])
                            if results:
                                all_results.extend(results)
                            
                            # Check if there are more pages for this postal code
                            total_pages = data.get("totalNumPages", 0)
                            if total_pages > page:
                                page += 1
                                break # Exit retry loop to fetch next page
                            else:
                                return all_results # Finished all pages
                        
                        elif response.status == 429 or response.status >= 500:
                            wait_time = (2 ** attempt) + 1
                            await asyncio.sleep(wait_time)
                            continue
                        
                        return all_results # Stop for 400/404
                except Exception:
                    await asyncio.sleep((2 ** attempt) + 1)
            else:
                # If we exhausted 4 attempts on a single page
                break
                
        return all_results

async def process_range(start, end):
    pcodes = [f"{p:06d}" for p in range(start, end + 1)]
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    results = []
    # Limit internal TCP connections to match our concurrency
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_pcode(p, session, semaphore) for p in pcodes]
        
        count = 0
        total = len(tasks)
        # Use as_completed to trigger progress updates as data arrives
        for f in asyncio.as_completed(tasks):
            res = await f
            results.append(res)
            count += 1
            
            # Print every 6,500 requests (~5.2 mins at ~21 RPS)
            if count % 1000 == 0 or count == total:
                print(f"[{start:06d}-{end:06d}] Progress: {count:,}/{total:,} ({count/total*100:.1f}%)")
    
    # Flatten list of lists into a single list of building dicts
    flattened = [b for sublist in results for b in sublist]
    if not flattened: return None
    
    df = pd.DataFrame(flattened)
    # Ensure Lat/Lon are floats for DuckDB range query compatibility
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    
    # Create GeoParquet with spatial metadata
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326"
    )
    return gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int, help="Start of postal code range")
    parser.add_argument("end", type=int, help="End of postal code range")
    args = parser.parse_args()

    print(f"🚀 Starting Stealth Scrape: {args.start:06d} to {args.end:06d}")
    
    gdf = asyncio.run(process_range(args.start, args.end))
    
    if gdf is not None and not gdf.empty:
        fname = f"addresses_{args.start:06d}_{args.end:06d}.parquet"
        gdf.to_parquet(fname, index=False)
        
        # Upload chunk to Hugging Face subfolder
        api = HfApi()
        api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=f"chunks/{fname}",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"✅ Successfully uploaded {fname} ({len(gdf)} records)")
    else:
        print("⚠️ No valid data found in this range.")
