"""
Model Training, Validation, and Benchmarking Pipeline.
Executes an Out-of-Time (OOT) temporal split (Train: Jan-Aug, Val: Sep-Oct),
benchmarks baselines against XGBoost, LightGBM, CatBoost, and Blended Ensembles,
and outputs performance metrics to scorer_results/.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    OOT_SPLIT_DATE,
    TARGET_COL,
    RESULTS_DIR,
)
from src.features import FreightFeaturePipeline
from src.models import (
    EquipmentBaseline,
    RidgeModel,
    RatePerMileWrapper,
    build_xgboost_model,
    build_lightgbm_model,
    build_catboost_model,
    BlendedEnsemble,
    compute_metrics,
)


def run_training_and_evaluation() -> dict[str, dict[str, float]]:
    print("=" * 75)
    print("      SPOTTER FREIGHT RATE ML PIPELINE: OUT-OF-TIME VALIDATION")
    print("=" * 75)

    # 1. Load Data
    raw_train = pd.read_csv(TRAIN_DATA_PATH)
    raw_val = pd.read_csv(VALIDATION_DATA_PATH)
    raw_train["date"] = pd.to_datetime(raw_train["date"])

    # 2. Temporal Out-of-Time Split
    split_dt = pd.to_datetime(OOT_SPLIT_DATE)
    train_split = raw_train[raw_train["date"] < split_dt].copy()
    val_split = raw_train[raw_train["date"] >= split_dt].copy()

    print(f"Total Labeled Dev Records: {len(raw_train):,}")
    print(f"OOT Training Set (Jan 1 - Aug 31): {len(train_split):,} rows ({len(train_split)/len(raw_train)*100:.1f}%)")
    print(f"OOT Validation Set (Sep 1 - Oct 31): {len(val_split):,} rows ({len(val_split)/len(raw_train)*100:.1f}%)")
    print("-" * 75)

    # 3. Fit Preprocessing Pipeline strictly on Training split
    pipeline = FreightFeaturePipeline()
    pipeline.fit(train_split, raw_val)

    # 4. Transform Datasets
    print("Transforming spatial, temporal, physical, and market features...")
    df_train_feat = pipeline.transform(train_split)
    df_val_feat = pipeline.transform(val_split)

    feature_cols = pipeline.get_feature_columns()
    print(f"Extracted {len(feature_cols)} engineered features for modeling.")

    X_train = df_train_feat[feature_cols].values
    y_train = df_train_feat[TARGET_COL].values
    dist_train = df_train_feat["distance"].values

    X_val = df_val_feat[feature_cols].values
    y_val = df_val_feat[TARGET_COL].values
    dist_val = df_val_feat["distance"].values

    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}

    # Model 1: Naive Equipment RPM Baseline
    print("\n[1/6] Training Naive Equipment RPM Baseline...")
    base_model = EquipmentBaseline()
    base_model.fit(train_split)
    pred_base = base_model.predict(val_split)
    results["Naive Baseline (Eq RPM)"] = compute_metrics(y_val, pred_base)
    predictions["Naive Baseline"] = pred_base

    # Model 2: Ridge Regression (RPM formulation)
    print("[2/6] Training Ridge Regression (RPM formulation)...")
    ridge_wrapper = RatePerMileWrapper(RidgeModel(alpha=10.0))
    ridge_wrapper.fit(X_train, y_train, dist_train)
    pred_ridge = ridge_wrapper.predict(X_val, dist_val)
    results["Ridge Regression (RPM)"] = compute_metrics(y_val, pred_ridge)
    predictions["Ridge"] = pred_ridge

    # Model 3: XGBoost (RPM formulation)
    print("[3/6] Training XGBoost Regressor (RPM formulation)...")
    xgb_wrapper = RatePerMileWrapper(build_xgboost_model())
    xgb_wrapper.fit(X_train, y_train, dist_train)
    pred_xgb = xgb_wrapper.predict(X_val, dist_val)
    results["XGBoost (RPM)"] = compute_metrics(y_val, pred_xgb)
    predictions["XGBoost"] = pred_xgb

    # Model 4: LightGBM (RPM formulation)
    print("[4/6] Training LightGBM Regressor (RPM formulation)...")
    lgb_wrapper = RatePerMileWrapper(build_lightgbm_model())
    lgb_wrapper.fit(X_train, y_train, dist_train)
    pred_lgb = lgb_wrapper.predict(X_val, dist_val)
    results["LightGBM (RPM)"] = compute_metrics(y_val, pred_lgb)
    predictions["LightGBM"] = pred_lgb

    # Model 5: CatBoost (RPM formulation)
    print("[5/6] Training CatBoost Regressor (RPM formulation)...")
    cat_wrapper = RatePerMileWrapper(build_catboost_model())
    cat_wrapper.fit(X_train, y_train, dist_train)
    pred_cat = cat_wrapper.predict(X_val, dist_val)
    results["CatBoost (RPM)"] = compute_metrics(y_val, pred_cat)
    predictions["CatBoost"] = pred_cat

    # Model 6: Blended Ensemble (XGBoost + LightGBM + CatBoost)
    print("[6/6] Building Blended Ensemble (XGBoost + LightGBM + CatBoost)...")
    ensemble = BlendedEnsemble(xgb_wrapper, lgb_wrapper, cat_wrapper, weights=(0.15, 0.20, 0.65))
    pred_ens = ensemble.predict(X_val, dist_val)
    results["Blended Ensemble"] = compute_metrics(y_val, pred_ens)
    predictions["Blended Ensemble"] = pred_ens

    # Display Results Comparison Table
    print("\n" + "=" * 75)
    print("               MODEL BENCHMARK COMPARISON TABLE (OOT VALIDATION)")
    print("=" * 75)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    print("=" * 75)

    # Save Results to JSON
    metrics_path = RESULTS_DIR / "validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved validation metrics JSON: {metrics_path}")

    # Plot Model Performance Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    models = list(results.keys())
    maes = [results[m]["MAE"] for m in models]
    r2s = [results[m]["R2"] for m in models]

    colors = ["#9E9E9E", "#5C6BC0", "#26A69A", "#42A5F5", "#AB47BC", "#E91E63"]
    bars1 = axes[0].barh(models, maes, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Out-of-Time MAE ($) - Lower is Better", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean Absolute Error ($)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    for bar, val in zip(bars1, maes):
        axes[0].text(val + 5, bar.get_y() + bar.get_height()/2, f"${val:.1f}", va="center", fontsize=9, fontweight="bold")

    bars2 = axes[1].barh(models, r2s, color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_title("Out-of-Time R² Score - Higher is Better", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("R² Score")
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    for bar, val in zip(bars2, r2s):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = RESULTS_DIR / "model_benchmark_comparison.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved benchmark comparison chart: {chart_path}")

    return results


if __name__ == "__main__":
    run_training_and_evaluation()
