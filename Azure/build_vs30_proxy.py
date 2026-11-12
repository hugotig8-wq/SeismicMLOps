import srtm
import pandas as pd
import numpy as np
import config
from pathlib import Path
import time

def main():
    print("Iniciando cálculo de Vs30 por proxy topográfico...")
    print("Descargando datos de elevación SRTM (esto puede tardar unos minutos)...")
    
    # Definir rutas relativas desde la carpeta Azure
    base_dir = Path("data")
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    input_parquet = processed_dir / "xgboost_earthquakes_cfm_dataset.parquet"
    output_csv = processed_dir / "vs30_grid.csv"
    output_parquet = processed_dir / "xgboost_earthquakes_vs30.parquet"
    
    try:
        geo_elevation_data = srtm.get_data()
    except Exception as e:
        print(f"Error al inicializar SRTM: {e}")
        return

    # Crear la malla usando los parámetros de config.py
    lat_step = (config.LAT_MAX - config.LAT_MIN) / config.GRID_ROWS
    lon_step = (config.LON_MAX - config.LON_MIN) / config.GRID_COLS
    
    results = []
    
    # Distancia aproximada de 1 km en grados para calcular el gradiente (pendiente)
    dlat = 1.0 / 111.32
    lon_center = (config.LON_MIN + config.LON_MAX) / 2
    dlon = 1.0 / (111.32 * np.cos(np.radians(lon_center)))
    
    for i in range(config.GRID_ROWS):
        for j in range(config.GRID_COLS):
            lat = config.LAT_MIN + (i + 0.5) * lat_step
            lon = config.LON_MIN + (j + 0.5) * lon_step
            
            try:
                # Obtener elevaciones (centro, 1km al norte, 1km al este)
                z0 = geo_elevation_data.get_elevation(lat, lon) or 0.0
                z_north = geo_elevation_data.get_elevation(lat + dlat, lon) or 0.0
                z_east = geo_elevation_data.get_elevation(lat, lon + dlon) or 0.0
                
                # Calcular gradiente (pendiente en m/m)
                dz_dy = (z_north - z0) / 1000.0
                dz_dx = (z_east - z0) / 1000.0
                
                slope_m_m = np.sqrt(dz_dx**2 + dz_dy**2)
                slope_deg = np.degrees(np.arctan(slope_m_m))
                
                # Fórmula de Allen & Wald (2009) para Vs30
                slope_deg_safe = max(slope_deg, 0.1)
                if slope_deg_safe < 3.5:
                    vs30 = 10**2.2
                elif slope_deg_safe > 35:
                    vs30 = 10**3.4
                else:
                    vs30 = 10**(2.2 + 0.4 * np.log10(slope_deg_safe))
                    
                results.append({
                    "grid_i": i, 
                    "grid_j": j, 
                    "vs30_proxy": vs30, 
                    "slope_deg": slope_deg
                })
            except Exception as e:
                # Si falla la descarga de un punto (ej. océano), ponemos roca dura por defecto
                results.append({"grid_i": i, "grid_j": j, "vs30_proxy": 760.0, "slope_deg": 0.0})
        
        print(f"Fila {i+1}/{config.GRID_ROWS} procesada...")
        time.sleep(0.5) # Pausa para no saturar el servidor de datos
            
    df_vs30 = pd.DataFrame(results)
    df_vs30.to_csv(output_csv, index=False)
    print(f"Datos de la malla guardados en: {output_csv}")
    
    # Enriquecer el Parquet existente
    if input_parquet.exists():
        print(f"Uniendo Vs30 al dataset principal ({input_parquet})...")
        df_eq = pd.read_parquet(input_parquet)
        
        if "grid_i" in df_eq.columns and "grid_j" in df_eq.columns:
            df_merged = df_eq.merge(df_vs30[["grid_i", "grid_j", "vs30_proxy"]], on=["grid_i", "grid_j"], how="left")
            df_merged["vs30_proxy"] = df_merged["vs30_proxy"].fillna(760.0) # Roca dura por defecto
            
            df_merged.to_parquet(output_parquet, index=False)
            print(f"¡Éxito! Nuevo dataset guardado en: {output_parquet}")
        else:
            print("Aviso: El parquet original no tiene las columnas grid_i/grid_j.")
    else:
        print(f"Error: No se encontró el archivo original en {input_parquet}")

if __name__ == "__main__":
    main()
