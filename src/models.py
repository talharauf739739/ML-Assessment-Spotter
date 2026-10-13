"""
Model architectures, benchmarking classes, and Rate-Per-Mile wrappers.
Includes Baselines, Ridge, XGBoost, LightGBM, CatBoost, and Blended Ensembles.
"""
from __future__ import annotations

from typing import Protocol
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.config import RANDOM_SEED


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes comprehensive regression metrics: MAE, RMSE, WAPE, and R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ensure strictly positive predictions (clip at $1.00)
    y_pred = np.clip(y_pred, 1.0, None)

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    wape = float(np.sum(np.abs(errors)) / (np.sum(y_true) + 1e-9) * 100.0)
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE (%)": round(wape, 2),
        "R2": round(r2, 4),
    }


class RatePerMileWrapper:
    """
    Wraps an underlying regressor to model Rate-Per-Mile (posted_rate / distance)
    rather than total posted_rate directly. At inference time, multiplies predicted
    RPM by shipment distance, guaranteeing positive rate predictions.
    """

    def __init__(self, base_estimator) -> None:
        self.base_estimator = base_estimator

    def fit(self, X: pd.DataFrame | np.ndarray, y_rate: np.ndarray, distance: np.ndarray) -> RatePerMileWrapper:
        y_rate = np.asarray(y_rate, dtype=float)
        distance = np.asarray(distance, dtype=float)
        y_rpm = y_rate / np.clip(distance, 1.0, None)
        self.base_estimator.fit(X, y_rpm)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, distance: np.ndarray) -> np.ndarray:
        distance = np.asarray(distance, dtype=float)
        pred_rpm = self.base_estimator.predict(X)
        pred_rpm = np.clip(pred_rpm, 0.20, None)  # Floor RPM at $0.20/mile
        pred_rate = pred_rpm * distance
        return np.clip(pred_rate, 10.0, None)


class EquipmentBaseline:
    """Naive baseline predicting historical mean rate-per-mile by equipment type."""

    def __init__(self) -> None:
        self.rpm_lookup: dict[str, float] = {}
        self.global_rpm: float = 2.22

    def fit(self, df: pd.DataFrame) -> EquipmentBaseline:
        df = df.copy()
        df["rpm"] = df["posted_rate"] / df["distance"]
        self.rpm_lookup = df.groupby("equipment")["rpm"].mean().to_dict()
        self.global_rpm = float(df["rpm"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            rpm = self.rpm_lookup.get(row.get("equipment", "Dry Van"), self.global_rpm)
            preds.append(rpm * float(row["distance"]))
        return np.array(preds)


class RidgeModel:
    """Standardized regularized linear regression model."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=RANDOM_SEED)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeModel:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def build_xgboost_model() -> xgb.XGBRegressor:
    """Builds optimized XGBoost Regressor."""
    return xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_lightgbm_model() -> lgb.LGBMRegressor:
    """Builds optimized LightGBM Regressor."""
    return lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost_model() -> CatBoostRegressor:
    """Builds optimized CatBoost Regressor."""
    return CatBoostRegressor(
        iterations=700,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_SEED,
        verbose=0,
        thread_count=-1,
    )


class BlendedEnsemble:
    """Blends predictions from XGBoost, LightGBM, and CatBoost models."""

    def __init__(self, xgb_model, lgb_model, cat_model, weights: tuple[float, float, float] = (0.15, 0.20, 0.65)) -> None:
        self.xgb = xgb_model
        self.lgb = lgb_model
        self.cat = cat_model
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray, distance: np.ndarray) -> BlendedEnsemble:
        self.xgb.fit(X, y, distance)
        self.lgb.fit(X, y, distance)
        self.cat.fit(X, y, distance)
        return self

    def predict(self, X: np.ndarray, distance: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb.predict(X, distance)
        p_lgb = self.lgb.predict(X, distance)
        p_cat = self.cat.predict(X, distance)
        w1, w2, w3 = self.weights
        return w1 * p_xgb + w2 * p_lgb + w3 * p_cat
