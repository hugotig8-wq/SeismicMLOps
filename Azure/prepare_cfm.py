# prepare_cfm.py
from __future__ import annotations
import sys
import zipfile
from pathlib import Path
import pandas as pd
import re

try:
    from pyproj import Transformer
except ImportError:
    print("ERROR: falta pyproj. Instálalo con: pip install pyproj")
    sys.exit(1)

# CONFIGURACIÓN DE RUTAS (Ajustada a config.py)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CFM_DIR = DATA_DIR / "CFM"
CFM_EXTRACTED_DIR = CFM_DIR / "extracted"
OUTPUT_DIR = DATA_DIR

METADATA_OUTPUT = CFM_DIR / "processed" / "cfm_metadata_clean.csv"
VERTICES_OUTPUT = CFM_DIR / "processed" / "cfm_vertices_1000m.csv"
TRIANGLES_OUTPUT = CFM_DIR / "processed" / "cfm_triangles_1000m.csv"
REPORT_OUTPUT = CFM_DIR / "processed" / "cfm_quality_report.txt"

CFM_CRS = "EPSG:26711"
EARTHQUAKE_CRS = "EPSG:4326"

def find_zip_file() -> Path | None:
    if not CFM_DIR.exists():
        CFM_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(CFM_DIR.glob("*.zip"))
    if not zip_files:
        return None
    # Buscar CFM 6.1 preferentemente
    cfm_zips = [p for p in zip_files if "cfm" in p.name.lower()]
    return cfm_zips[0] if cfm_zips else zip_files[0]

def find_metadata_file() -> Path | None:
    candidates = list(CFM_DIR.rglob("*Metadata*.xlsx"))
    return candidates[0] if candidates else None

def find_1000m_directory() -> Path | None:
    candidates = [p for p in CFM_DIR.rglob("1000m") if p.is_dir() and "preferred" in str(p).lower()]
    return candidates[0] if candidates else None

def extract_cfm_zip(zip_path: Path) -> Path:
    if CFM_EXTRACTED_DIR.exists() and list(CFM_EXTRACTED_DIR.rglob("*")):
        return CFM_EXTRACTED_DIR
    CFM_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extrayendo {zip_path.name} en {CFM_EXTRACTED_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(CFM_EXTRACTED_DIR)
    return CFM_EXTRACTED_DIR

def prepare_cfm_directory() -> None:
    CFM_DIR.mkdir(parents=True, exist_ok=True)
    if find_metadata_file() and find_1000m_directory():
        return
    zip_path = find_zip_file()
    if not zip_path:
        raise FileNotFoundError(f"No se encontró ZIP en {CFM_DIR}. Coloca el ZIP de CFM 6.1 ahí.")
    extract_cfm_zip(zip_path)

VRTX_PATTERN = re.compile(r"^\s*VRTX\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")
TRGL_PATTERN = re.compile(r"^\s*TRGL\s+(\d+)\s+(\d+)\s+(\d+)")

def parse_tsurf(ts_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    vertices, triangles = [], []
    triangle_id = 0
    with ts_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            v_match = VRTX_PATTERN.match(line)
            if v_match:
                vertices.append({"vertex_id": int(v_match.group(1)), "x": float(v_match.group(2)), "y": float(v_match.group(3)), "z": float(v_match.group(4))})
                continue
            t_match = TRGL_PATTERN.match(line)
            if t_match:
                triangle_id += 1
                triangles.append({"triangle_id": triangle_id, "v1": int(t_match.group(1)), "v2": int(t_match.group(2)), "v3": int(t_match.group(3))})
    return pd.DataFrame(vertices), pd.DataFrame(triangles)

def convert_vertices_to_wgs84(vertices: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs(CFM_CRS, EARTHQUAKE_CRS, always_xy=True)
    lon, lat = transformer.transform(vertices["x"].to_numpy(), vertices["y"].to_numpy())
    result = vertices.copy()
    result["longitude"] = lon
    result["latitude"] = lat
    result["elevation_or_depth_m"] = result["z"]
    result["depth_km"] = -result["z"] / 1000.0
    return result

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(metadata_path)
    preferred_sheet = next((s for s in excel.sheet_names if "fault" in s.lower() or "metadata" in s.lower()), excel.sheet_names[0])
    metadata = pd.read_excel(metadata_path, sheet_name=preferred_sheet).dropna(how="all").reset_index(drop=True)
    metadata.columns = [str(c).strip() for c in metadata.columns]
    
    # Estandarización de columnas para CFM 6.0 y 6.1
    col_map = {}
    for col in metadata.columns:
        c_low = col.lower()
        if "fault object name" in c_low or "fault name" in c_low: col_map[col] = "Fault_Name"
        elif "wavgstrike" in c_low or "average strike" in c_low: col_map[col] = "Strike"
        elif "wavgdip" in c_low or "average dip" in c_low: col_map[col] = "Dip"
        elif "totalarea" in c_low or "area" in c_low: col_map[col] = "Area_km2"
        elif "slip sense" in c_low: col_map[col] = "Slip_Sense"
    
    metadata = metadata.rename(columns=col_map)
    return metadata

def process_surfaces(ts_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_files = sorted(ts_directory.glob("*.ts"))
    all_vertices, all_triangles = [], []
    print(f"Procesando {len(ts_files)} superficies...")
    for index, ts_path in enumerate(ts_files, start=1):
        fault_name = ts_path.stem
        v, t = parse_tsurf(ts_path)
        if v.empty: continue
        v["fault_name"] = fault_name
        t["fault_name"] = fault_name
        all_vertices.append(convert_vertices_to_wgs84(v))
        all_triangles.append(t)
    return pd.concat(all_vertices, ignore_index=True), pd.concat(all_triangles, ignore_index=True)

def main():
    print("--- PREPARANDO CFM ---")
    prepare_cfm_directory()
    meta_path = find_metadata_file()
    ts_dir = find_1000m_directory()
    
    if not meta_path or not ts_dir:
        raise FileNotFoundError("No se encontraron archivos tras la extracción. Revisa el ZIP.")

    print(f"Metadata: {meta_path}")
    metadata = load_metadata(meta_path)
    vertices, triangles = process_surfaces(ts_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(METADATA_OUTPUT, index=False)
    vertices.to_csv(VERTICES_OUTPUT, index=False)
    triangles.to_csv(TRIANGLES_OUTPUT, index=False)
    print(f"Guardado: {VERTICES_OUTPUT}")
    print(f"Guardado: {METADATA_OUTPUT}")
    print("--- COMPLETADO ---")

if __name__ == "__main__":
    main()
