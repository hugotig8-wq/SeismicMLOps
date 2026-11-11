"""Machine learning model training and inference module."""

from typing import Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

import config

def get_calibrated_model(base_model):
    """Retorna el modelo de calibración asegurando compatibilidad de versiones de scikit-learn."""
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model), 
            method=config.CALIBRATION_METHOD
        )
    except ImportError:
        return CalibratedClassifierCV(
            estimator=base_model, 
            method=config.CALIBRATION_METHOD, 
            cv="prefit"
        )

def get_base_model(model_type: str = config.MODEL_TYPE, scale_pos_weight: float = config.SCALE_POS_WEIGHT):
    """Instancia el modelo según la configuración de config.py"""
    params = config.MODEL_CONFIGS.get(model_type, {}).copy()
    
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
        
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, scale_pos_weight=scale_pos_weight)
        
    elif model_type == "random_forest":
        return RandomForestClassifier(**params, class_weight="balanced")
        
    else:
        raise ValueError(f"Modelo '{model_type}' no soportado.")

def train_and_calibrate_model(X_train, y_train, X_val, y_val, scale_pos_weight: float = None):
    if scale_pos_weight is None:
        # Calcular automáticamente según la proporción de negativos/positivos en Train
        num_pos = max(int(y_train.sum()), 1)
        num_neg = len(y_train) - num_pos
        scale_pos_weight = float(num_neg / num_pos)

    base_model = get_base_model(config.MODEL_TYPE, scale_pos_weight=scale_pos_weight)
    base_model.fit(X_train, y_train)
    
    calibrated_model = get_calibrated_model(base_model)
    calibrated_model.fit(X_val, y_val)
    
    # Guardar modelo entrenado automáticamente usando config.MODEL_DIR
    model_path = config.MODEL_DIR / f"calibrated_{config.MODEL_TYPE}_model.joblib"
    joblib.dump(calibrated_model, model_path)
    
    return calibrated_model
    
