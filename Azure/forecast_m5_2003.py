"""forecast_m5_2003.py - Pipeline de Entrenamiento, Calibración y Evaluación."""

import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score, confusion_matrix

import config
from models import train_and_calibrate_model
from features import (
    assign_grid_indices,
    add_cfm_features,
    add_vs30_features,
    prepare_forecast_data, # <--- NUEVA FUNCIÓN
    grid_to_latlon
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_forecast_pipeline():
    # 1. Cargar catálogo raw desde el PARQUET con Vs30 ya integrado
    try:
        if config.USE_VS30:
            logging.info(f"Cargando catálogo desde {config.VS30_DATA_PATH}...")
            df_raw = pd.read_parquet(config.VS30_DATA_PATH)
        elif config.USE_SCEC_CFM:
            logging.info(f"Cargando catálogo desde {config.CFM_DATA_PATH}...")
            df_raw = pd.read_parquet(config.CFM_DATA_PATH)
    except:
        loggin.info("Error al cargar parquet.")
    # 2. Mapear celdas (por si el parquet no las trae explícitas)
    df_mapped = assign_grid_indices(df_raw)
    
    # 3. Construcción de datasets de Train y Test (Target temporal + features históricas)
    logging.info("Construyendo features temporales y asignando targets sin data leakage...")
    df_train_grid, df_test_grid = prepare_forecast_data(
        df_mapped, 
        train_cutoff_year=2003, 
        val_cutoff_year=2013, 
        target_mag=config.TARGET_MAGNITUDE
    )
    
    # 4. Unir Vs30 (y CFM si algún día lo reactivas)
    # Nota: Como leemos del parquet con Vs30, la columna ya existe, pero 
    # por seguridad hacemos el merge desde el CSV para garantizar que se cruce bien.
    if config.USE_VS30:
        logging.info("Enriqueciendo datasets con VS30 proxy...")
        df_train_grid = add_vs30_features(df_train_grid, config.VS30CSV_DATA_PATH)
        df_test_grid = add_vs30_features(df_test_grid, config.VS30CSV_DATA_PATH)

    if config.USE_SCEC_CFM:
        logging.info("Enriqueciendo datasets con características SCEC CFM...")
        df_train_grid = add_cfm_features(df_train_grid, config.CFM_DATA_PATH)
        df_test_grid = add_cfm_features(df_test_grid, config.CFM_DATA_PATH)
   
    ###Este paso a continuación no sirvió, parece que 2 columnas colineales le dieron mejor y más puntos de corte de árbol para elegir y mejora todas las métricas considerablemente.
    ## 6. Eliminar redundancia: dejar solo vs30_log (eliminamos vs30_proxy crudo)
    ## XGBoost sufre con variables colineales (una es el log de la otra).
    #logging.info("Eliminando 'vs30_proxy' para dejar solo 'vs30_log'...")
    #cols_to_drop_redundant = ["vs30_proxy"]

    #df_train_grid = df_train_grid.drop(columns=[c for c in cols_to_drop_redundant if c in df_train_grid.columns])
    #df_test_grid = df_test_grid.drop(columns=[c for c in cols_to_drop_redundant if c in df_test_grid.columns])

    # 7. Separación de matriz de características (X) y objetivo (y)
    drop_cols = ["target", "grid_i", "grid_j", "max_past_magnitude"]
    # ... (el resto del código de split sigue igual)

    X_train_full = df_train_grid.drop(columns=[c for c in drop_cols if c in df_train_grid.columns])
    y_train_full = df_train_grid["target"]

    X_test = df_test_grid.drop(columns=[c for c in drop_cols if c in df_test_grid.columns])
    y_test = df_test_grid["target"]

    # --- NUEVOS PRINTS DE DIAGNÓSTICO ---
    logging.info(f"Positivos en Train Completo: {y_train_full.sum()} de {len(y_train_full)} celdas")
    logging.info(f"Positivos en Test Completo: {y_test.sum()} de {len(y_test)} celdas")
    # ------------------------------------

    # Asignar mayor proporción a Validación...
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.35, random_state=42, stratify=y_train_full
    )

    # --- NUEVOS PRINTS DE DIAGNÓSTICO 2 ---
    logging.info(f"Positivos en X_train: {y_train.sum()}, Positivos en y_val: {y_val.sum()}")
    # --------------------------------------

    logging.info(f"Dataset Split completado: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # 6. Entrenamiento de XGBoost y Calibración Sigmoide
    logging.info("Entrenando modelo XGBoost y aplicando calibración sigmoide...")
    calibrated_model = train_and_calibrate_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    # 7. Evaluación en conjunto de Test
    logging.info("Evaluando el modelo calibrado sobre el conjunto de test...")
    
    # En lugar de:
    # y_probs = calibrated_model.predict_proba(X_test)[:, 1]

    # Usa esto:
    probs = calibrated_model.predict_proba(X_test)
    if probs.shape[1] == 2:
        y_probs = probs[:, 1]
    else:
        # Si el modelo solo devuelve 1 columna, es porque solo vio la clase 0
        y_probs = np.zeros(len(X_test))
        logging.warning("¡Atención! El modelo solo devolvió 1 clase. Probabilidades puestas a 0.")

    if len(set(y_test)) > 1:
        auc_score = float(roc_auc_score(y_test, y_probs))
    else:
        auc_score = 0.5
        logging.warning("El conjunto de prueba solo contiene una clase. Se asigna ROC-AUC por defecto de 0.5.")

    brier_score = float(brier_score_loss(y_test, y_probs))
    max_prob = float(y_probs.max())

    # --- CÁLCULO DE MÉTRICAS EXTENDIDAS ---

    # 1. Top-K Recall (K = 10% de las celdas con mayor probabilidad)
    k = max(1, int(len(y_probs) * 0.10))
    top_k_indices = np.argsort(y_probs)[-k:]

    # Extraemos los targets reales (y_test) de esas Top-K celdas
    y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    top_k_positives = np.sum(y_test_array[top_k_indices] == 1)
    total_positives = np.sum(y_test_array == 1)
    top_k_recall = top_k_positives / total_positives if total_positives > 0 else 0.0

    # 2. Métricas y Matriz de Confusión para múltiples umbrales (0.1, 0.2, 0.3, 0.4, 0.5)
    umbrales = [0.1, 0.2, 0.3, 0.4, 0.5]
    metricas_umbral = {}

    for u in umbrales:
        y_pred = (y_probs >= u).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_array, y_pred, labels=[0, 1]).ravel()
    
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
        metricas_umbral[str(u)] = {
            "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "TPR": round(tpr, 4), "FPR": round(fpr, 4)
        }

# --------------------------------------

    logging.info(f"Test ROC-AUC: {auc_score:.4f}")
    logging.info(f"Test Brier Score: {brier_score:.4f}")
    logging.info(f"Máxima Probabilidad Calibrada: {max_prob:.4f}")
    logging.info(f"Top-K recall: {top_k_recall: 4f}")
    for u in umbrales:
        logging.info(f"Matriz de confusión de umbral {u}: {metricas_umbral[str(u)]}.")

    # 8. Generar y Exportar Mapa de Riesgo
    X_test_map = X_test.copy()
    X_test_map["grid_i"] = df_test_grid.loc[X_test.index, "grid_i"]
    X_test_map["grid_j"] = df_test_grid.loc[X_test.index, "grid_j"]
    X_test_map["risk_probability"] = y_probs

    X_test_map = grid_to_latlon(X_test_map)

    # Reordenar columnas
    cols_order = ["google_maps_url", "latitude", "longitude", "risk_probability", "seismic_rate"] + [
        c for c in X_test_map.columns if c not in ["google_maps_url", "latitude", "longitude", "risk_probability", "seismic_rate"]
    ]
    X_test_map = X_test_map[cols_order]

    X_test_map.to_csv(config.RISK_MAP_SAVE_PATH, index=False)
    logging.info(f"Mapa de riesgo calibrado guardado en {config.RISK_MAP_SAVE_PATH}")

    # Guardar reporte de métricas
    metrics_payload = {
        "model_type": config.MODEL_TYPE,
        "calibration_method": config.CALIBRATION_METHOD,
        "use_cfm": config.USE_SCEC_CFM,
        "use_vs30": config.USE_VS30,
        "roc_auc": round(auc_score, 4),
        "brier_score": round(brier_score, 4),
        "max_calibrated_probability": round(max_prob, 4),
        "n_features": X_train.shape[1],
        "features": X_train.columns.tolist(),
        "used_parquet": config.OUTPUT_PARQUET,
        "tt_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_k_recall_k": k,
        "top_k_recall_value": round(top_k_recall, 4),
        "umbral_metrics": metricas_umbral
    }

    with open(config.METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    logging.info(f"Métricas del modelo exportadas a {config.METRICS_SAVE_PATH}")

if __name__ == "__main__":
    run_forecast_pipeline()
