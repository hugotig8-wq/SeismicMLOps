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

def assign_grid_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna los índices de celda (grid_i, grid_j) a las coordenadas del dataset."""
    df = df.copy()
    
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise KeyError("El DataFrame debe contener las columnas 'latitude' y 'longitude'.")
        
    lat_step = (config.LAT_MAX - config.LAT_MIN) / config.GRID_ROWS
    lon_step = (config.LON_MAX - config.LON_MIN) / config.GRID_COLS

    if "grid_i" not in df.columns or "grid_j" not in df.columns:
        df["grid_i"] = ((df["latitude"] - config.LAT_MIN) / lat_step).astype(int)
        df["grid_j"] = ((df["longitude"] - config.LON_MIN) / lon_step).astype(int)
    
        df["grid_i"] = df["grid_i"].clip(0, config.GRID_ROWS - 1)
        df["grid_j"] = df["grid_j"].clip(0, config.GRID_COLS - 1)
    
    return df

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
    for col in cols_to_agg:
        df_cfm[col] = pd.to_numeric(df_cfm[col], errors="coerce")
        
    cfm_summary = df_cfm.groupby(["grid_i", "grid_j"])[cols_to_agg].mean().reset_index()

    df_merged = pd.merge(df, cfm_summary, on=["grid_i", "grid_j"], how="left")

    for col in cols_to_agg:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    return df_merged

def add_vs30_features(df_grid: pd.DataFrame, vs30_path: str = None) -> pd.DataFrame:
    """Integra la característica Vs30 desde el CSV y aplica logaritmo."""
    if vs30_path is None:
        vs30_path = config.VS30CSV_DATA_PATH
        
    df = df_grid.copy()
    
    try:
        df_vs30 = pd.read_csv(vs30_path)
    except Exception as e:
        print(f"Error cargando Vs30: {e}")
        return df

    # Asegurar que ambas tablas tengan grid_i y grid_j
    if "grid_i" not in df_vs30.columns or "grid_j" not in df_vs30.columns:
        df_vs30 = assign_grid_indices(df_vs30)

    df_vs30["vs30_proxy"] = pd.to_numeric(df_vs30["vs30_proxy"], errors="coerce").fillna(760.0)
    df_vs30["vs30_log"] = np.log1p(df_vs30["vs30_proxy"])

    # Nos quedamos solo con las columnas necesarias para el merge
    df_vs30_subset = df_vs30[["grid_i", "grid_j", "vs30_proxy", "vs30_log"]].drop_duplicates(subset=["grid_i", "grid_j"])

    df_merged = pd.merge(df, df_vs30_subset, on=["grid_i", "grid_j"], how="left")
    df_merged["vs30_proxy"] = df_merged["vs30_proxy"].fillna(760.0)
    df_merged["vs30_log"] = df_merged["vs30_log"].fillna(np.log1p(760.0))
    
    print(f"      [Vs30] Característica añadida. Rango: "
          f"{df_merged['vs30_proxy'].min():.1f} - {df_merged['vs30_proxy'].max():.1f} m/s. "
          f"(Log1p: {df_merged['vs30_log'].min():.1f} - {df_merged['vs30_log'].max():.1f})")
    
    return df_merged

def prepare_forecast_data(df_events: pd.DataFrame, train_cutoff_year: int = 2003, 
                          val_cutoff_year: int = 2013, target_mag: float = config.TARGET_MAGNITUDE) -> tuple:
    """
    Crea las matrices X e y con targets temporales (sin data leakage) y 
    agrega las features sísmicas por celda para el entrenamiento del modelo.
    """
    df = df_events.copy()
    
    # 1. Asegurar columna 'year'
    if "year" not in df.columns and "time" in df.columns:
        df["year"] = pd.to_datetime(df["time"], format="mixed", errors="coerce").dt.year

    # 2. Calcular features históricas estáticas usando solo el período de Train (<= 2003)
    df_train_period = df[df["year"] <= train_cutoff_year]
    
    # Agrupamos por celda las estadísticas del catálogo histórico
    df_grid_features = df_train_period.groupby(["grid_i", "grid_j"]).agg(
        seismic_rate=("magnitude", "count"),
        max_past_magnitude=("magnitude", lambda x: x.max() if len(x) > 0 else 0.0),
        mean_magnitude=("magnitude", lambda x: x.mean() if len(x) > 0 else 0.0),
        std_magnitude=("magnitude", lambda x: x.std() if len(x) > 1 else 0.0)
    ).reset_index()

    # Rellenar NaNs
    df_grid_features["max_past_magnitude"] = df_grid_features["max_past_magnitude"].fillna(0.0)
    df_grid_features["mean_magnitude"] = df_grid_features["mean_magnitude"].fillna(0.0)
    df_grid_features["std_magnitude"] = df_grid_features["std_magnitude"].fillna(0.0)

    # 3. Calcular b_value por celda (basado en historial)
    b_values = []
    for _, row in df_grid_features.iterrows():
        cell_mags = df_train_period[(df_train_period["grid_i"] == row["grid_i"]) & 
                                    (df_train_period["grid_j"] == row["grid_j"])]["magnitude"]
        b_values.append(calculate_b_value(cell_mags, mc=config.MIN_MAGNITUDE))
    df_grid_features["b_value"] = b_values

    # 4. Definir los targets futuros (lo que queremos predecir)
    df_train_targets = df[(df["year"] > train_cutoff_year) & (df["year"] <= val_cutoff_year)]
    df_test_targets = df[df["year"] > val_cutoff_year]

    train_m_pairs = df_train_targets[df_train_targets["magnitude"] >= target_mag][["grid_i", "grid_j"]].drop_duplicates()
    test_m_pairs = df_test_targets[df_test_targets["magnitude"] >= target_mag][["grid_i", "grid_j"]].drop_duplicates()

    # 5. Asignar target 1/0 a la malla
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
