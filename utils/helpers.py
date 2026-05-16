import pandas as pd
import time
import geopandas as gpd
import quackosm as qosm 
from pathlib import Path

def save_sliver(df: pd.DataFrame, dataset_name: str, path: str):
    df.to_parquet(Path(f"{path}/{dataset_name}.parquet"), compression="zstd", index=False)



def parse_poi(input_file, output_file, pbf_file, radius):
    start_time = time.time()

    print("1. Reading map data from local PBF file using QuackOSM...")

    # Define the exact tags we want at the parser level
    tags_filter = {
        'amenity': ['school', 'hospital', 'marketplace'],
        'highway': ['bus_stop'],
        'tourism': ['attraction', 'hotel', 'guest_house'],
        'shop': ['supermarket']
    }

    pois_gdf = qosm.convert_pbf_to_geodataframe(
        pbf_file,  # use parameter, not global
        tags_filter=tags_filter
    )
    print(f"-> Extracted {len(pois_gdf)} total POIs from the map.")

    print("\n2. Loading your outlet coordinates...")
    coords = pd.read_parquet(input_file)  # use parameter

    outlets_gdf = gpd.GeoDataFrame(
        coords,
        geometry=gpd.points_from_xy(coords['Longitude'], coords['Latitude']),
        crs="EPSG:4326"
    )

    print(f"\n3. Projecting and buffering geometries to {radius}m...")  # use parameter

    outlets_metric = outlets_gdf.to_crs("EPSG:3857")
    pois_metric = pois_gdf.to_crs("EPSG:3857")

    outlets_metric['geometry'] = outlets_metric.geometry.buffer(radius)  # use parameter

    print("\n4. Performing spatial join... (The fast part!)")
    joined = gpd.sjoin(pois_metric, outlets_metric, how="inner", predicate="intersects")

    print("\n5. Aggregating and formatting results...")
    joined['poi_type'] = None

    if 'amenity' in joined.columns:
        joined.loc[joined['amenity'] == 'school', 'poi_type'] = f'poi_school_{radius}m'
        joined.loc[joined['amenity'] == 'hospital', 'poi_type'] = f'poi_hospital_{radius}m'
        joined.loc[joined['amenity'] == 'marketplace', 'poi_type'] = f'poi_market_{radius}m'

    if 'highway' in joined.columns:
        joined.loc[joined['highway'] == 'bus_stop', 'poi_type'] = f'poi_bus_stop_{radius}m'

    if 'shop' in joined.columns:
        joined.loc[joined['shop'] == 'supermarket', 'poi_type'] = f'poi_supermarket_{radius}m'

    if 'tourism' in joined.columns:
        joined.loc[joined['tourism'].isin(['attraction', 'hotel', 'guest_house']), 'poi_type'] = f'poi_tourism_{radius}m'

    counts = joined.dropna(subset=['poi_type']).groupby(['Outlet_ID', 'poi_type']).size().reset_index(name='count')

    final_df = counts.pivot(index='Outlet_ID', columns='poi_type', values='count').fillna(0).astype(int).reset_index()

    final_df = pd.merge(coords[['Outlet_ID']], final_df, on='Outlet_ID', how='left').fillna(0)

    for col in final_df.columns:
        if col != 'Outlet_ID':
            final_df[col] = final_df[col].astype(int)

    final_df.to_parquet(output_file, index=False)  # use parameter

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {len(coords)} outlets in {elapsed:.2f} seconds.")
    print(f"Saved to: {output_file}")