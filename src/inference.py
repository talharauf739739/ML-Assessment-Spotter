"""
Batch Inference & Submission Pipeline.
Trains final models on the complete 48,000 development loads,
generates validation_predictions.csv, completes december_chart_inputs.csv,
and triggers score.py to produce scorer_results/candidate_december.png.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    DECEMBER_CHART_INPUTS_PATH,
    VALIDATION_PREDICTIONS_PATH,
    PROJECT_ROOT,
    TARGET_COL,
    RESULTS_DIR,
)
from src.features import FreightFeaturePipeline
from src.models import (
    RatePerMileWrapper,
    build_xgboost_model,
    build_lightgbm_model,
    build_catboost_model,
    BlendedEnsemble,
)


def run_full_inference() -> None:
    print("=" * 75)
    print("        SPOTTER FREIGHT RATE ML PIPELINE: FINAL BATCH INFERENCE")
    print("=" * 75)

    # 1. Load Datasets
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VALIDATION_DATA_PATH)
    dec_df = pd.read_csv(DECEMBER_CHART_INPUTS_PATH)

    print(f"Full Development Set: {len(train_df):,} rows")
    print(f"Final Validation Set: {len(val_df):,} loads")
    print(f"December Chart Input: {len(dec_df):,} daily loads")

    # 2. Fit Feature Pipeline on Full Development Set
    print("\nFitting feature pipeline on full historical data (Jan - Oct 2025)...")
    pipeline = FreightFeaturePipeline()
    pipeline.fit(train_df, val_df)

    # 3. Transform Datasets
    print("Extracting engineered features for all datasets...")
    train_feat = pipeline.transform(train_df)
    val_feat = pipeline.transform(val_df)
    dec_feat = pipeline.transform(dec_df)

    feature_cols = pipeline.get_feature_columns()
    X_train = train_feat[feature_cols].values
    y_train = train_feat[TARGET_COL].values
    dist_train = train_feat["distance"].values

    X_val = val_feat[feature_cols].values
    dist_val = val_feat["distance"].values

    X_dec = dec_feat[feature_cols].values
    dist_dec = dec_feat["distance"].values

    # 4. Train Final Blended Ensemble on Full Data
    print("\nTraining final production Blended Ensemble (XGBoost + LightGBM + CatBoost)...")
    xgb_wrapper = RatePerMileWrapper(build_xgboost_model())
    lgb_wrapper = RatePerMileWrapper(build_lightgbm_model())
    cat_wrapper = RatePerMileWrapper(build_catboost_model())
    ensemble = BlendedEnsemble(xgb_wrapper, lgb_wrapper, cat_wrapper, weights=(0.35, 0.40, 0.25))

    ensemble.fit(X_train, y_train, dist_train)
    print("Model training complete.")

    # 5. Predict on Validation Set
    print("\nGenerating predictions for validation.csv...")
    val_preds = ensemble.predict(X_val, dist_val)
    val_preds = np.round(val_preds, 2)

    submission_df = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": val_preds,
    })

    # Save to validation_predictions.csv in project root
    submission_df.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    print(f"Saved final validation predictions to: {VALIDATION_PREDICTIONS_PATH}")
    print(f"Validation predictions summary: Mean=${val_preds.mean():.2f}, Min=${val_preds.min():.2f}, Max=${val_preds.max():.2f}")

    # 6. Predict on December Scenario
    print("\nGenerating predictions for data/december_chart_inputs.csv...")
    dec_preds = ensemble.predict(X_dec, dist_dec)
    dec_preds = np.round(dec_preds, 2)

    dec_completed = dec_df.copy()
    dec_completed["predicted_rate"] = dec_preds

    # Ensure required columns and column order
    cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_completed = dec_completed[cols]

    dec_completed.to_csv(DECEMBER_CHART_INPUTS_PATH, index=False)
    print(f"Saved completed December chart predictions to: {DECEMBER_CHART_INPUTS_PATH}")
    print(f"December predictions summary: Mean=${dec_preds.mean():.2f}, Min=${dec_preds.min():.2f}, Max=${dec_preds.max():.2f}")

    # 7. Execute score.py Scorer
    print("\nExecuting Spotter official score.py script...")
    score_script = PROJECT_ROOT / "score.py"
    cmd = [
        sys.executable,
        str(score_script),
        "--predictions",
        str(VALIDATION_PREDICTIONS_PATH),
        "--december-predictions",
        str(DECEMBER_CHART_INPUTS_PATH),
        "--output-dir",
        str(RESULTS_DIR),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR executing score.py:", result.stderr)
        raise SystemExit(result.returncode)

    print("=" * 75)
    print("ALL VALIDATION & SCORER CHECKS PASSED SUCCESSFULLY!")
    print(f"Generated chart available at: {RESULTS_DIR / 'candidate_december.png'}")
    print("=" * 75)


if __name__ == "__main__":
    run_full_inference()
