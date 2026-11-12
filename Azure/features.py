
"""Spatial binning and seismic feature engineering module."""

import numpy as np
import pandas as pd
import config

def calculate_b_value(magnitudes: pd.Series, mc: float = 1.4) -> float:
    """Calcula el valor b de Gutenberg-Richter mediante Máxima Verosimilitud (Aki, 1965)."""
    mags = magnitudes[magnitudes >= mc]
    if len(mags) < 5:
        return 1.0
    mean_m = mags.mean()
    b = (1.0 / (mean_m - (mc - 0.05))) * np.log10(np.e)
    return float(b)

def add_cfm_features(df_grid: pd.DataFrame, cfm_path: str = None) -> pd.DataFrame:
    """Integra características geométricas y geológicas 3D del CFM a la malla espacial."""
    if cfm_path is None:
        cfm_path = config.CFM_DATA_PATH
        
    df = df_grid.copy()
    
    try:
        df_cfm = pd.read_parquet(cfm_path)
    except Exception:
        return df

    if "grid_i" not in df_cfm.columns or "grid_j" not in df_cfm.columns:
        df_cfm = assign_grid_indices(df_cfm)

    numeric_cols = [
        "distance_to_nearest_fault_km",
        "nearest_fault_vertex_depth_km",
        "nearest_fault_strike",
        "nearest_fault_dip",
        "nearest_fault_area_km2",
        "fault_count_5km",
        "fault_count_10km",
        "fault_count_20km",
        "fault_density_10km"
    ]

    cols_to_agg = [c for c in numeric_cols if c in df_cfm.columns]
    # FIX: Forzar conversión a numérico por si el Excel trae símbolos extraños
    for col in cols_to_agg:
        df_cfm[col] = pd.to_numeric(df_cfm[col], errors="coerce")
        
    cfm_summary = df_cfm.groupby(["grid_i", "grid_j"])[cols_to_agg].mean().reset_index()

    df_merged = pd.merge(df, cfm_summary, on=["grid_i", "grid_j"], how="left")

    for col in cols_to_agg:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    return df_merged

def assign_grid_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna los índices de celda (grid_i, grid_j) a las coordenadas del dataset."""
    df = df.copy()
    
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise KeyError("El DataFrame debe contener las columnas 'latitude' y 'longitude'.")
        
    lat_step = (config.LAT_MAX - config.LAT_MIN) / config.GRID_ROWS
    lon_step = (config.LON_MAX - config.LON_MIN) / config.GRID_COLS
    
    df["grid_i"] = ((df["latitude"] - config.LAT_MIN) / lat_step).astype(int)
    df["grid_j"] = ((df["longitude"] - config.LON_MIN) / lon_step).astype(int)
    
    df["grid_i"] = df["grid_i"].clip(0, config.GRID_ROWS - 1)
    df["grid_j"] = df["grid_j"].clip(0, config.GRID_COLS - 1)
    
    return df

def build_grid_features(df_events: pd.DataFrame, cutoff_year: int = 2003) -> pd.DataFrame:
    """Agrupa el catálogo por celda e incluye la tasa de aceleración temporal."""
    df = df_events.copy()
    
    if "grid_i" not in df.columns or "grid_j" not in df.columns:
        df = assign_grid_indices(df)
        
    if "year" not in df.columns and "time" in df.columns:
        df["year"] = pd.to_datetime(df["time"], format="mixed", errors="coerce").dt.year

    if cutoff_year is not None and "year" in df.columns:
        df = df[df["year"] <= cutoff_year]

    # 1. Agregaciones estándar históricas
    grid_df = df.groupby(["grid_i", "grid_j"]).agg(
        seismic_rate=("magnitude", "count"),
        max_past_magnitude=("magnitude", lambda x: x.max() if len(x) > 0 else 0.0),
        mean_magnitude=("magnitude", lambda x: x.mean() if len(x) > 0 else 0.0),
        std_magnitude=("magnitude", lambda x: x.std() if len(x) > 1 else 0.0)
    ).reset_index()

    grid_df["max_past_magnitude"] = grid_df["max_past_magnitude"].fillna(0.0)
    grid_df["mean_magnitude"] = grid_df["mean_magnitude"].fillna(0.0)
    grid_df["std_magnitude"] = grid_df["std_magnitude"].fillna(0.0)

    # 2. Tasa de aceleración reciente (últimos 5 años pre-corte)
    recent_start_year = cutoff_year - 4
    df_recent = df[df["year"] >= recent_start_year].groupby(["grid_i", "grid_j"])["magnitude"].count().reset_index()
    df_recent.columns = ["grid_i", "grid_j", "recent_seismic_rate"]

    grid_df = pd.merge(grid_df, df_recent, on=["grid_i", "grid_j"], how="left")
    grid_df["recent_seismic_rate"] = grid_df["recent_seismic_rate"].fillna(0.0)
    grid_df["seismic_acceleration"] = grid_df["recent_seismic_rate"] / (grid_df["seismic_rate"] + 1e-5)

    # 3. Cálculo del b-value por celda
    b_values = []
    for _, row in grid_df.iterrows():
        cell_mags = df[(df["grid_i"] == row["grid_i"]) & (df["grid_j"] == row["grid_j"])]["magnitude"]
        b_val = calculate_b_value(cell_mags, mc=config.MIN_MAGNITUDE)
        b_values.append(b_val)
    
    grid_df["b_value"] = b_values

    return grid_df

def create_temporal_split_datasets(df_mapped: pd.DataFrame, 
                                   train_cutoff_year: int = 2003, 
                                   val_cutoff_year: int = 2013, 
                                   target_mag: float = config.TARGET_MAGNITUDE):
    """Crea matrices de Train y Test aplicando la magnitud objetivo configurada."""
    df = df_mapped.copy()
    if "year" not in df.columns and "time" in df.columns:
        df["year"] = pd.to_datetime(df["time"], format="mixed", errors="coerce").dt.year

    df_grid_features = build_grid_features(df, cutoff_year=train_cutoff_year)

    df_train_period = df[(df["year"] > train_cutoff_year) & (df["year"] <= val_cutoff_year)]
    df_test_period = df[df["year"] > val_cutoff_year]

    train_m_pairs = df_train_period[df_train_period["magnitude"] >= target_mag][["grid_i", "grid_j"]].drop_duplicates()
    test_m_pairs = df_test_period[df_test_period["magnitude"] >= target_mag][["grid_i", "grid_j"]].drop_duplicates()

    grid_index = pd.MultiIndex.from_frame(df_grid_features[["grid_i", "grid_j"]])
    
    train_index = pd.MultiIndex.from_frame(train_m_pairs) if not train_m_pairs.empty else pd.MultiIndex(levels=[[],[]], codes=[[],[]])
    test_index = pd.MultiIndex.from_frame(test_m_pairs) if not test_m_pairs.empty else pd.MultiIndex(levels=[[],[]], codes=[[],[]])

    df_train = df_grid_features.copy()
    df_train["target"] = grid_index.isin(train_index).astype(int)

    df_test = df_grid_features.copy()
    df_test["target"] = grid_index.isin(test_index).astype(int)

    return df_train, df_test

def grid_to_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los índices de la celda a coordenadas geográficas."""
    df = df.copy()
    
    lat_step = (config.LAT_MAX - config.LAT_MIN) / config.GRID_ROWS
    lon_step = (config.LON_MAX - config.LON_MIN) / config.GRID_COLS
    
    df["latitude"] = config.LAT_MIN + (df["grid_i"] + 0.5) * lat_step
    df["longitude"] = config.LON_MIN + (df["grid_j"] + 0.5) * lon_step
    
    df["google_maps_url"] = df.apply(
        lambda r: f"https://www.google.com/maps?q={r['latitude']:.4f},{r['longitude']:.4f}", axis=1
    )
    
    return df

def add_vs30_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la característica de Vs30 y aplica transformaciones para
    mejorar el rendimiento de XGBoost.
    """
    if "vs30_proxy" not in df.columns:
        raise ValueError(
            "La columna 'vs30_proxy' no está en el DataFrame. "
            "Verifica que config.cfm_data_path apunta al archivo xgboost_cfm_vs30.parquet"
        )
    
    # 1. Tratamiento de nulos (por si acaso, aunque ya los rellenamos a 760)
    df["vs30_proxy"] = df["vs30_proxy"].fillna(760.0)
    
    # 2. Transformación logarítmica (opcional pero muy recomendada)
    # Ayuda a XGBoost a manejar la escala geológica de forma más suave.
    # Lo guardamos como una nueva columna para no perder el valor original.
    df["vs30_log"] = np.log1p(df["vs30_proxy"])
    
    print(f"      [Vs30] Característica añadida. Rango original: "
          f"{df['vs30_proxy'].min():.1f} - {df['vs30_proxy'].max():.1f} m/s")
    
    return df




    
