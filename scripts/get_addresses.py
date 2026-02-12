import asyncio
import aiohttp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from huggingface_hub import HfApi
import os
import argparse

# ULTRA-STEALTH CONFIGURATION
CONCURRENT_REQUESTS = 10 
TOKEN = os.getenv("ONEMAP_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "gisfun/spatial-datasets"

async def fetch_pcode(pcode, session, semaphore):
    async with semaphore:
        page = 1
        all_results = []
        headers = {} # {"Authorization": TOKEN}

        while True:
            # 1.5s delay per page request to be absolutely safe
            await asyncio.sleep(1.5) 
            
            # CORRECT ONEMAP URL
            url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={pcode}&returnGeom=Y&getAddrDetails=Y&pageNum={page}"
            
            success = False
            for attempt in range(4):
                try:
                    async with session.get(url, headers=headers, timeout=20) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get("results", [])
                            if results:
                                all_results.extend(results)
                            
                            # Pagination logic
                            if data.get("totalNumPages", 0) > page:
                                page += 1
                                success = True 
                                break 
                            else:
                                return all_results 
                        
                        elif response.status == 429 or response.status >= 500:
                            wait_time = (2 ** attempt) + 2
                            print(f"⚠️ [PCODE {pcode}] Status {response.status}. Backing off {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"🛑 [PCODE {pcode}] Permanent Error {response.status}. Skipping.")
                            return all_results 
                except Exception as e:
                    wait_time = (2 ** attempt) + 2
                    print(f"❌ [PCODE {pcode}] Error: {str(e)[:100]}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            
            if not success: 
                break
        return all_results

async def process_range(start, end):
    pcodes = [f"{p:06d}" for p in range(start, end + 1)]
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    results = []
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_pcode(p, session, semaphore) for p in pcodes]
        
        count = 0
        total = len(tasks)
        for f in asyncio.as_completed(tasks):
            res = await f
            results.append(res)
            count += 1
            
            # Print every 50 for quick feedback in logs
            if count % 50 == 0 or count == total:
                print(f"[{start:06d}-{end:06d}] Progress: {count:,}/{total:,} ({count/total*100:.1f}%)")
    
    flattened = [b for sublist in results for b in sublist]
    if not flattened: return None
    
    df = pd.DataFrame(flattened)
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326"
    )
    return gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    args = parser.parse_args()
    
    print(f"🚀 Starting Ultra-Stealth Scrape: {args.start:06d} to {args.end:06d}")
    gdf = asyncio.run(process_range(args.start, args.end))
    
    if gdf is not None and not gdf.empty:
        fname = f"addresses_{args.start:06d}_{args.end:06d}.parquet"
        gdf.to_parquet(fname, index=False)
        HfApi().upload_file(
            path_or_fileobj=fname, 
            path_in_repo=f"chunks/{fname}",
            repo_id=REPO_ID, 
            repo_type="dataset", 
            token=HF_TOKEN
        )
