# config.py
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. RUTAS DEL PROYECTO (CORREGIDAS)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Definimos la carpeta base de datos (puedes cambiarla a la ruta absoluta de tu PC)
DATA_DIR = PROJECT_ROOT / "data" 
DATA_PATH = DATA_DIR / "earthquakes.csv"

CFM_DATA_PATH = DATA_DIR / "xgboost_earthquakes_cfm_dataset.parquet"
CVM_DATA_PATH = DATA_DIR / "scec_cvm_data.csv"
CTM_DATA_PATH = DATA_DIR / "scec_ctm_data.csv"

PROCESSED_DIR = PROJECT_ROOT / "processed"
RISKMAPS_DIR = PROJECT_ROOT / "riskMaps"
METRICS_DIR = PROJECT_ROOT / "metrics"
MODEL_DIR = PROJECT_ROOT / "training" / "models"

for directory in [DATA_DIR, PROCESSED_DIR, METRICS_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RISK_MAP_SAVE_PATH = RISKMAPS_DIR / f"risk_map_calibrated_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv"
METRICS_SAVE_PATH = METRICS_DIR / f"forecast_metrics{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json"

# ============================================================
# 2. PARÁMETROS GEOGRÁFICOS Y DE LA MALLA
# ============================================================
LAT_MIN = 32.0
LAT_MAX = 36.0
LON_MIN = -120.0
LON_MAX = -115.0

GRID_ROWS = 18
GRID_COLS = 18
CELL_KM = 10.0

# ============================================================
# 3. PARÁMETROS SISMOLÓGICOS Y MODELADO HIGH-ACCURACY
# ============================================================
MIN_MAGNITUDE = 1.4
TARGET_MAGNITUDE = 4.5

MODEL_TYPE = "xgboost"

MODEL_CONFIGS = {
    "xgboost": {
        "n_estimators": 50,
        "max_depth": 1,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }
}

SCALE_POS_WEIGHT = 1.5
CALIBRATION_METHOD = "sigmoid"

# ============================================================
# 4. SWITCHES DE MODELOS SCEC
# ============================================================
USE_SCEC_CFM = True
USE_SCEC_CTM = False
USE_SCEC_CVM = False
