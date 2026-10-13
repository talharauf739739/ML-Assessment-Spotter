"""
Exploratory Data Analysis (EDA) & Data Quality Audit Module.
Generates statistical diagnostics, checks for missing data patterns,
and saves key exploratory visualizations to scorer_results/.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    RESULTS_DIR,
)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads training development and validation datasets."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VALIDATION_DATA_PATH)
    train_df["date"] = pd.to_datetime(train_df["date"])
    val_df["date"] = pd.to_datetime(val_df["date"])
    return train_df, val_df


def run_data_quality_audit(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Performs missing value audits, dimension checks, and distribution checks."""
    print("=" * 70)
    print("             SPOTTER FREIGHT DATA QUALITY AUDIT")
    print("=" * 70)
    print(f"Development Set Shape : {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"Development Date Range: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Validation Set Shape  : {val_df.shape[0]:,} rows x {val_df.shape[1]} columns")
    print(f"Validation Date Range : {val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')}")
    
    print("\n--- Missing Values Audit ---")
    train_missing = train_df.isnull().sum()[train_df.isnull().sum() > 0]
    val_missing = val_df.isnull().sum()[val_df.isnull().sum() > 0]
    print(f"Train Missing Features:\n{train_missing}")
    print(f"\nValidation Missing Features:\n{val_missing}")

    print("\n--- Target Variable ('posted_rate') Summary ---")
    print(train_df["posted_rate"].describe())

    train_df["rate_per_mile"] = train_df["posted_rate"] / train_df["distance"]
    print("\n--- Rate Per Mile (posted_rate / distance) Summary ---")
    print(train_df["rate_per_mile"].describe())

    print("\n--- Rate Per Mile by Equipment Type ---")
    rpm_by_eq = train_df.groupby("equipment")["rate_per_mile"].agg(["count", "mean", "median", "std"])
    print(rpm_by_eq)

    print("\n--- Correlation with Target ('posted_rate') ---")
    num_cols = ["distance", "weight", "market_index", "quote_signal", "posted_rate", "rate_per_mile"]
    corr = train_df[num_cols].corr()
    print(corr[["posted_rate", "rate_per_mile"]].round(4))
    print("=" * 70)

    return {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_missing": train_missing.to_dict(),
        "val_missing": val_missing.to_dict(),
    }


def generate_eda_plots(train_df: pd.DataFrame) -> None:
    """Creates publication-ready EDA plots for documentation and the final report."""
    train_df = train_df.copy()
    train_df["rate_per_mile"] = train_df["posted_rate"] / train_df["distance"]

    # 1. Distribution of Distance & Rate Per Mile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    
    axes[0].hist(train_df["distance"], bins=40, color="#1D4E89", edgecolor="white", alpha=0.85)
    axes[0].set_title("Linehaul Distance Distribution (miles)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Distance (miles)")
    axes[0].set_ylabel("Load Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].hist(train_df["rate_per_mile"], bins=40, color="#0072BB", edgecolor="white", alpha=0.85, range=(0, 6))
    axes[1].axvline(train_df["rate_per_mile"].median(), color="#D64045", linestyle="--", linewidth=2, label=f"Median: ${train_df['rate_per_mile'].median():.2f}/mi")
    axes[1].set_title("Rate Per Mile Distribution ($/mile)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Rate Per Mile ($)")
    axes[1].set_ylabel("Load Count")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    dist_plot_path = RESULTS_DIR / "eda_distributions.png"
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"Saved distribution plot: {dist_plot_path}")

    # 2. Rate per Mile by Equipment Type Boxplot / Violin
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    equipments = ["Dry Van", "Flatbed", "Reefer"]
    data = [train_df[train_df["equipment"] == eq]["rate_per_mile"].dropna() for eq in equipments]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=equipments, showfliers=False)
    colors = ["#4D908E", "#F9844A", "#F94144"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    ax.set_title("Rate Per Mile Across Equipment Types (Excl. Outliers)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rate Per Mile ($)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    eq_plot_path = RESULTS_DIR / "eda_equipment_rates.png"
    plt.savefig(eq_plot_path)
    plt.close()
    print(f"Saved equipment rate plot: {eq_plot_path}")

    # 3. Temporal Trend: Daily Average Rate Per Mile Across 2025
    daily = train_df.groupby("date")["rate_per_mile"].mean()
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
    ax.plot(daily.index, daily.values, color="#277DA1", linewidth=1.8, label="Daily Mean RPM")
    ax.plot(daily.index, daily.rolling(7).mean(), color="#F94144", linewidth=2.4, label="7-Day Rolling Trend")
    ax.set_title("Temporal Market Trajectory: Daily Rate Per Mile (Jan - Oct 2025)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Rate Per Mile ($)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    temporal_plot_path = RESULTS_DIR / "eda_temporal_trend.png"
    plt.savefig(temporal_plot_path)
    plt.close()
    print(f"Saved temporal trend plot: {temporal_plot_path}")


def main() -> None:
    train_df, val_df = load_datasets()
    run_data_quality_audit(train_df, val_df)
    generate_eda_plots(train_df)


if __name__ == "__main__":
    main()
