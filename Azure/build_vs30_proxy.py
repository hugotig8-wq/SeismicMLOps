"""
build_vs30_proxy.py

Ubicación: /home/humberto/SeismicMLOps/Azure/build_vs30_proxy.py

Enriquece el parquet del catálogo sísmico con la característica `vs30_proxy`:

    input : Azure/data/processed/xgboost_earthquakes_cfm_dataset.parquet
    vs30  : Azure/data/processed/vs30_grid.csv
    output: xgboost_cfm_vs30.parquet #Luego config.OUTPUT_PARQUET

Si el parquet de entrada no trae `grid_i` / `grid_j`, se calculan con
`assign_grid_indices()` de features.py (misma lógica que el pipeline
principal). Los huecos de Vs30 se rellenan con 760 m/s (roca firme).
"""

from pathlib import Path
import sys
import pandas as pd
import config

# ---------------------------------------------------------------------------
# Rutas (script ubicado en Azure/)
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent          # .../Azure
DATA_DIR    = SCRIPT_DIR / "data" / "processed"

INPUT_PARQUET  = DATA_DIR / "xgboost_earthquakes_cfm_dataset.parquet"
VS30_GRID_FILE = DATA_DIR / "vs30_grid.csv"
OUTPUT_PARQUET = DATA_DIR / "xgboost_cfm_vs30.parquet" #Luego con su archivo generador y fecha config.OUTPUT_PARQUET

VS30_DEFAULT = 760.0  # m/s — fallback para celdas sin dato

# config.py y features.py viven en SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import *                       # noqa: F401,F403
from features import assign_grid_indices   # noqa: E402


# ---------------------------------------------------------------------------
# Carga de la tabla Vs30
# ---------------------------------------------------------------------------
def load_vs30_grid(path: Path) -> pd.DataFrame:
    """
    Carga la tabla Vs30 y garantiza que tenga (grid_i, grid_j, vs30_proxy).
    Acepta CSV (por defecto) o parquet.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la tabla Vs30: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado para Vs30: {path.suffix}")

    # Caso A: ya trae grid_i / grid_j
    if {"grid_i", "grid_j"}.issubset(df.columns):
        pass
    # Caso B: sólo lat/lon → calculamos índices con la misma función del pipeline
    elif {"latitude", "longitude"}.issubset(df.columns):
        print("      Vs30 sin grid_i/grid_j → calculando con assign_grid_indices()")
        df = assign_grid_indices(df)
    else:
        raise ValueError(
            "La tabla Vs30 debe traer (grid_i, grid_j) o (latitude, longitude). "
            f"Columnas disponibles: {list(df.columns)}"
        )

    if "vs30_proxy" not in df.columns:
        raise ValueError(
            "La tabla Vs30 no contiene 'vs30_proxy'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    return (
        df[["grid_i", "grid_j", "vs30_proxy"]]
        .drop_duplicates(subset=["grid_i", "grid_j"], keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def build_vs30_proxy() -> None:
    print(f"[i] INPUT_PARQUET  -> {INPUT_PARQUET}")
    print(f"[i] VS30_GRID_FILE -> {VS30_GRID_FILE}")
    print(f"[i] OUTPUT_PARQUET -> {OUTPUT_PARQUET}")

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {INPUT_PARQUET}")

    # [1/4] Leer parquet principal
    print(f"[1/4] Leyendo parquet: {INPUT_PARQUET.name}")
    df_eq = pd.read_parquet(INPUT_PARQUET)
    print(f"      Filas: {len(df_eq):,} | Columnas: {len(df_eq.columns)}")

    # [2/4] Garantizar grid_i / grid_j reutilizando features.py
    if "grid_i" not in df_eq.columns or "grid_j" not in df_eq.columns:
        print("[2/4] Calculando grid_i / grid_j con assign_grid_indices() ...")
        df_eq = assign_grid_indices(df_eq)
    else:
        print("[2/4] grid_i / grid_j ya presentes en el parquet.")

    # [3/4] Cargar Vs30
    print(f"[3/4] Cargando Vs30 desde {VS30_GRID_FILE.name} ...")
    df_vs30 = load_vs30_grid(VS30_GRID_FILE)
    print(f"      Celdas Vs30 disponibles: {len(df_vs30):,}")

    # [4/4] Merge + relleno
    print("[4/4] Merge y relleno de huecos ...")
    df_merged = df_eq.merge(
        df_vs30,
        on=["grid_i", "grid_j"],
        how="left",
        validate="many_to_one",
    )

    n_missing = int(df_merged["vs30_proxy"].isna().sum())
    if n_missing > 0:
        print(f"      Rellenando {n_missing:,} celdas sin Vs30 con {VS30_DEFAULT} m/s")
        df_merged["vs30_proxy"] = df_merged["vs30_proxy"].fillna(VS30_DEFAULT)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"[OK] Escrito: {OUTPUT_PARQUET}")
    print(f"     Filas finales: {len(df_merged):,} | Columnas: {len(df_merged.columns)}")


if __name__ == "__main__":
    build_vs30_proxy()
