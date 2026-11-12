"""
build_vs30_proxy.py

Enriquece el parquet del catálogo sísmico con la característica `vs30_proxy`.

Ubicación esperada:
    /home/humberto/SeismicMLOps/Azure/build_vs30_proxy.py

Busca config.py, features.py y los parquets tanto en Azure/ como en la
raíz del proyecto (SeismicMLOps/), para funcionar sin importar dónde
estén ubicados. Al arrancar imprime por consola qué ficheros ha usado.
"""

from pathlib import Path
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# 1) Localización robusta de config.py / features.py
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent   # .../SeismicMLOps/Azure
PROJECT_ROOT = SCRIPT_DIR.parent                 # .../SeismicMLOps

# Añadimos ambos directorios a sys.path (sin duplicar)
for _base in (SCRIPT_DIR, PROJECT_ROOT):
    if str(_base) not in sys.path:
        sys.path.insert(0, str(_base))

try:
    import config  # noqa: E402
except ImportError as exc:
    raise ImportError(
        f"No se pudo importar config.py. Buscado en:\n"
        f"  - {SCRIPT_DIR}\n"
        f"  - {PROJECT_ROOT}"
    ) from exc

try:
    from features import assign_grid_indices  # noqa: E402
except ImportError as exc:
    raise ImportError(
        f"No se pudo importar features.py (con assign_grid_indices). Buscado en:\n"
        f"  - {SCRIPT_DIR}\n"
        f"  - {PROJECT_ROOT}"
    ) from exc

print(f"[i] config.py   -> {config.__file__}")
print(f"[i] features.py -> {sys.modules['features'].__file__}")


# ---------------------------------------------------------------------------
# 2) Localización de los parquets (busca en Azure/data y raíz/data)
# ---------------------------------------------------------------------------
def _find_data_file(filename: str) -> Path:
    candidates = [
        SCRIPT_DIR / "data" / filename,
        PROJECT_ROOT / "data" / filename,
        SCRIPT_DIR / filename,
        PROJECT_ROOT / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # devuelve la primera aunque no exista → error claro


def _find_vs30_file() -> Path:
    """Busca el fichero Vs30 por nombre explícito y, si no, por glob."""
    explicit_names = ["vs30_grid.parquet", "vs30.parquet",
                      "vs30_grid.csv",     "vs30.csv"]
    for name in explicit_names:
        p = _find_data_file(name)
        if p.exists():
            return p
    # Búsqueda difusa: cualquier vs30*.parquet / *.csv
    search_dirs = [SCRIPT_DIR / "data", PROJECT_ROOT / "data",
                   SCRIPT_DIR, PROJECT_ROOT]
    for base in search_dirs:
        if base.is_dir():
            for pattern in ("vs30*.parquet", "vs30*.csv"):
                matches = sorted(base.glob(pattern))
                if matches:
                    return matches[0]
    # No encontrado → devolvemos el nombre esperado para el mensaje de error
    return SCRIPT_DIR / "data" / "vs30_grid.parquet"


INPUT_PARQUET  = _find_data_file("xgboost_earthquakes_cfm_dataset.parquet")
VS30_GRID_FILE = _find_vs30_file()
OUTPUT_PARQUET = INPUT_PARQUET.with_name(
    INPUT_PARQUET.stem + "_vs30.parquet"
)

VS30_DEFAULT = 760.0  # m/s — roca firme (fallback)


# ---------------------------------------------------------------------------
# 3) Carga de la tabla Vs30
# ---------------------------------------------------------------------------
def load_vs30_grid(path: Path) -> pd.DataFrame:
    """
    Carga la tabla Vs30 y garantiza que tenga (grid_i, grid_j, vs30_proxy).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la tabla Vs30. Esperada en: {path}\n"
            f"Si el fichero se llama distinto, renómbralo a 'vs30.parquet' "
            f"o 'vs30_grid.parquet' y colócalo en data/."
        )

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado para Vs30: {path.suffix}")

    # Caso A: ya trae grid_i / grid_j
    if {"grid_i", "grid_j"}.issubset(df.columns):
        pass
    # Caso B: trae lat/lon → calculamos índices con la misma función del pipeline
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
# 4) Pipeline principal
# ---------------------------------------------------------------------------
def build_vs30_proxy() -> None:
    print(f"[i] INPUT_PARQUET  -> {INPUT_PARQUET}")
    print(f"[i] VS30_GRID_FILE -> {VS30_GRID_FILE}")
    print(f"[i] OUTPUT_PARQUET -> {OUTPUT_PARQUET}")

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"No se encontró el parquet de entrada: {INPUT_PARQUET}"
        )

    # [1/4] Leer parquet principal
    print(f"[1/4] Leyendo parquet: {INPUT_PARQUET.name}")
    df_eq = pd.read_parquet(INPUT_PARQUET)
    print(f"      Filas: {len(df_eq):,} | Columnas: {len(df_eq.columns)}")

    # [2/4] Garantizar grid_i / grid_j
    if "grid_i" not in df_eq.columns or "grid_j" not in df_eq.columns:
        print("[2/4] Calculando grid_i / grid_j con assign_grid_indices() ...")
        df_eq = assign_grid_indices(df_eq)
    else:
        print("[2/4] grid_i / grid_j ya presentes en el parquet.")

    # [3/4] Cargar Vs30
    print(f"[3/4] Cargando Vs30 desde {VS30_GRID_FILE.name} ...")
    df_vs30 = load_vs30_grid(VS30_GRID_FILE)
    print(f"      Celdas Vs30 disponibles: {len(df_vs30):,}")

    # [4/4] Merge + relleno de huecos
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
