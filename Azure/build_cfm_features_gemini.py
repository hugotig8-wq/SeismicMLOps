# build_cfm_features_gemini.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

EARTHQUAKES_FILE = DATA_DIR / "earthquakes.csv"
CFM_VERTICES_FILE = DATA_DIR / "processed" / "cfm_vertices_1000m.csv"
CFM_METADATA_FILE = DATA_DIR / "processed" / "cfm_metadata_clean.csv"
OUTPUT_PARQUET = DATA_DIR / "processed" / "xgboost_earthquakes_cfm_dataset.parquet"

CFM_CRS = "EPSG:26711"
EARTHQUAKE_CRS = "EPSG:4326"
RADII_KM = [5.0, 10.0, 20.0]

def load_and_project_earthquakes() -> pd.DataFrame:
    df = pd.read_csv(EARTHQUAKES_FILE)
    transformer = Transformer.from_crs(EARTHQUAKE_CRS, CFM_CRS, always_xy=True)
    x_utm, y_utm = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    
    df["x_utm"] = x_utm
    df["y_utm"] = y_utm
    # Aseguramos el nombre de columna 'depth' (puede venir como 'Depth')
    depth_col = "depth" if "depth" in df.columns else "Depth"
    df["z_m"] = -df[depth_col] * 1000.0 
    return df

def main() -> None:
    print("Cargando datos...")
    eq_df = load_and_project_earthquakes()
    vertices_df = pd.read_csv(CFM_VERTICES_FILE)
    metadata_df = pd.read_csv(CFM_METADATA_FILE)

    print("Construyendo árbol KDTree 3D...")
    cfm_coords_3d = vertices_df[["x", "y", "z"]].to_numpy()
    tree_3d = cKDTree(cfm_coords_3d)

    eq_coords_3d = eq_df[["x_utm", "y_utm", "z_m"]].to_numpy()
    n_earthquakes = len(eq_df)

    print("Calculando distancias 3D...")
    distances_m, nearest_indices = tree_3d.query(eq_coords_3d, k=1)

    eq_df["distance_to_nearest_fault_km"] = distances_m / 1000.0
    eq_df["nearest_fault_name"] = vertices_df.iloc[nearest_indices]["fault_name"].values
    eq_df["nearest_fault_vertex_depth_km"] = vertices_df.iloc[nearest_indices]["depth_km"].values

    print("Calculando conteo de FALLAS ÚNICAS por lotes...")
    fault_names_array = pd.factorize(vertices_df["fault_name"])[0]
    batch_size = 1000

    for r_km in RADII_KM:
        r_m = r_km * 1000.0
        unique_counts = np.zeros(n_earthquakes, dtype=np.int32)
        print(f"  -> Procesando radio {int(r_km)} km...")
        for start_idx in range(0, n_earthquakes, batch_size):
            end_idx = min(start_idx + batch_size, n_earthquakes)
            batch_coords = eq_coords_3d[start_idx:end_idx]
            batch_neighbors = tree_3d.query_ball_point(batch_coords, r=r_m)
            
            unique_counts[start_idx:end_idx] = [
                len(set(fault_names_array[idxs])) if idxs else 0 for idxs in batch_neighbors
            ]
        eq_df[f"fault_count_{int(r_km)}km"] = unique_counts

    eq_df["fault_density_10km"] = eq_df["fault_count_10km"] / (np.pi * (10.0 ** 2))

    print("Uniendo metadata geológica...")
    # Seleccionamos usando los nombres estandarizados del prepare_cfm.py
    cols_to_merge = ["Fault_Name", "Strike", "Dip", "Area_km2", "Slip_Sense"]
    available_cols = [c for c in cols_to_merge if c in metadata_df.columns]
    metadata_sub = metadata_df[available_cols].drop_duplicates(subset=["Fault_Name"])

    eq_df = eq_df.merge(
        metadata_sub,
        left_on="nearest_fault_name",
        right_on="Fault_Name",
        how="left"
    )

    eq_df.rename(columns={
        "Strike": "nearest_fault_strike",
        "Dip": "nearest_fault_dip",
        "Area_km2": "nearest_fault_area_km2",
        "Slip_Sense": "nearest_fault_slip_sense"
    }, inplace=True)

    features_to_keep = [
        "earthquake_id", "latitude", "longitude", "depth", "magnitude",
        "distance_to_nearest_fault_km", "nearest_fault_name",
        "nearest_fault_vertex_depth_km", "nearest_fault_strike", 
        "nearest_fault_dip", "nearest_fault_area_km2", "nearest_fault_slip_sense",
        "fault_count_5km", "fault_count_10km", "fault_count_20km", "fault_density_10km"
    ]
    
    final_cols = [col for col in features_to_keep if col in eq_df.columns]
    final_dataset = eq_df[final_cols]
    final_dataset.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nDataset guardado en:\n{OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
