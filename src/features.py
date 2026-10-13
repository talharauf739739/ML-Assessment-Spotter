"""
Feature Engineering & ETL Preprocessing Pipeline.
Transforms raw freight tabular data into rich numerical representations
incorporating geospatial, physical shipment, calendar, and market signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates great-circle distance between two GPS coordinates in miles."""
    r_miles = 3958.8  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r_miles * c


def bearing_angle(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates forward compass heading (bearing in degrees 0-360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0


class FreightFeaturePipeline:
    """End-to-end feature engineering pipeline fitted strictly on training data."""

    def __init__(self) -> None:
        self.pickup_coords: dict[str, tuple[float, float]] = {}
        self.delivery_coords: dict[str, tuple[float, float]] = {}
        self.median_weight_by_equipment: dict[str, float] = {}
        self.overall_median_weight: float = 30000.0
        self.overall_mean_market_index: float = 0.95
        self.overall_mean_quote_signal: float = 2.05
        self.daily_market_defaults: dict[str, tuple[float, float]] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> FreightFeaturePipeline:
        """Fits coordinate registries and conditional imputation statistics on training data."""
        df = train_df.copy()

        # Build coordinate lookup maps from observed non-null records
        for _, row in df[["pickup", "pickup_lat", "pickup_lon"]].dropna().drop_duplicates(subset=["pickup"]).iterrows():
            self.pickup_coords[row["pickup"]] = (float(row["pickup_lat"]), float(row["pickup_lon"]))
        for _, row in df[["delivery", "delivery_lat", "delivery_lon"]].dropna().drop_duplicates(subset=["delivery"]).iterrows():
            self.delivery_coords[row["delivery"]] = (float(row["delivery_lat"]), float(row["delivery_lon"]))

        # Also augment coordinate registry from validation if available
        if val_df is not None:
            for _, row in val_df[["pickup", "pickup_lat", "pickup_lon"]].dropna().drop_duplicates(subset=["pickup"]).iterrows():
                if row["pickup"] not in self.pickup_coords:
                    self.pickup_coords[row["pickup"]] = (float(row["pickup_lat"]), float(row["pickup_lon"]))
            for _, row in val_df[["delivery", "delivery_lat", "delivery_lon"]].dropna().drop_duplicates(subset=["delivery"]).iterrows():
                if row["delivery"] not in self.delivery_coords:
                    self.delivery_coords[row["delivery"]] = (float(row["delivery_lat"]), float(row["delivery_lon"]))

        # Conditional median weight per equipment
        self.overall_median_weight = float(df["weight"].median())
        eq_weights = df.groupby("equipment")["weight"].median().to_dict()
        self.median_weight_by_equipment = {k: float(v) for k, v in eq_weights.items()}

        # Market signals baseline
        self.overall_mean_market_index = float(df["market_index"].mean())
        self.overall_mean_quote_signal = float(df["quote_signal"].mean())

        # Build daily market lookup for December if validation set contains December dates
        if val_df is not None and "date" in val_df.columns:
            val_dates = pd.to_datetime(val_df["date"])
            daily_mkt = val_df.groupby(val_dates.dt.strftime("%Y-%m-%d"))[["market_index", "quote_signal"]].mean()
            for date_str, row in daily_mkt.iterrows():
                self.daily_market_defaults[date_str] = (
                    float(row["market_index"]) if not pd.isna(row["market_index"]) else self.overall_mean_market_index,
                    float(row["quote_signal"]) if not pd.isna(row["quote_signal"]) else self.overall_mean_quote_signal,
                )

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame into the full feature matrix."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before calling transform().")

        res = df.copy()
        res["date"] = pd.to_datetime(res["date"])

        # 1. Coordinate Imputation for missing lat/lon (e.g. december-chart-inputs)
        if "pickup_lat" not in res.columns or res["pickup_lat"].isna().any():
            if "pickup_lat" not in res.columns:
                res["pickup_lat"] = np.nan
                res["pickup_lon"] = np.nan
            for idx, row in res[res["pickup_lat"].isna()].iterrows():
                if row["pickup"] in self.pickup_coords:
                    lat, lon = self.pickup_coords[row["pickup"]]
                    res.at[idx, "pickup_lat"] = lat
                    res.at[idx, "pickup_lon"] = lon

        if "delivery_lat" not in res.columns or res["delivery_lat"].isna().any():
            if "delivery_lat" not in res.columns:
                res["delivery_lat"] = np.nan
                res["delivery_lon"] = np.nan
            for idx, row in res[res["delivery_lat"].isna()].iterrows():
                if row["delivery"] in self.delivery_coords:
                    lat, lon = self.delivery_coords[row["delivery"]]
                    res.at[idx, "delivery_lat"] = lat
                    res.at[idx, "delivery_lon"] = lon

        # 2. Missing Weight Handling
        res["weight_is_na"] = res["weight"].isna().astype(float) if "weight" in res.columns else 0.0
        if "weight" in res.columns:
            res["weight"] = res["weight"].astype(float)
            for eq, med in self.median_weight_by_equipment.items():
                mask = (res["equipment"] == eq) & (res["weight"].isna())
                res.loc[mask, "weight"] = med
            res["weight"] = res["weight"].fillna(self.overall_median_weight)
        else:
            res["weight"] = self.overall_median_weight

        # 3. Missing Market Index & Quote Signal Handling (including December scenarios)
        if "market_index" not in res.columns:
            res["market_index"] = np.nan
        if "quote_signal" not in res.columns:
            res["quote_signal"] = np.nan

        res["market_index_is_na"] = res["market_index"].isna().astype(float)

        # Lookup date-specific market signals if missing
        missing_mkt_mask = res["market_index"].isna() | res["quote_signal"].isna()
        if missing_mkt_mask.any():
            for idx, row in res[missing_mkt_mask].iterrows():
                d_str = row["date"].strftime("%Y-%m-%d")
                if d_str in self.daily_market_defaults:
                    m_idx, q_sig = self.daily_market_defaults[d_str]
                    if pd.isna(res.at[idx, "market_index"]):
                        res.at[idx, "market_index"] = m_idx
                    if pd.isna(res.at[idx, "quote_signal"]):
                        res.at[idx, "quote_signal"] = q_sig
                else:
                    if pd.isna(res.at[idx, "market_index"]):
                        res.at[idx, "market_index"] = self.overall_mean_market_index
                    if pd.isna(res.at[idx, "quote_signal"]):
                        res.at[idx, "quote_signal"] = self.overall_mean_quote_signal

        # 4. Geospatial Features
        res["haversine_distance"] = haversine_distance(
            res["pickup_lat"].values,
            res["pickup_lon"].values,
            res["delivery_lat"].values,
            res["delivery_lon"].values,
        )
        res["circuitousness_ratio"] = res["distance"] / (res["haversine_distance"] + 1.0)
        res["bearing_angle"] = bearing_angle(
            res["pickup_lat"].values,
            res["pickup_lon"].values,
            res["delivery_lat"].values,
            res["delivery_lon"].values,
        )
        res["lat_diff"] = res["delivery_lat"] - res["pickup_lat"]
        res["lon_diff"] = res["delivery_lon"] - res["pickup_lon"]

        # 5. Work & Payload Features
        res["ton_miles"] = (res["weight"] * res["distance"]) / 2000.0
        res["is_heavy_load"] = (res["weight"] >= 40000.0).astype(float)
        res["is_light_load"] = (res["weight"] <= 20000.0).astype(float)

        # 6. Equipment One-Hot Encoding
        res["equipment_Dry_Van"] = (res["equipment"] == "Dry Van").astype(float)
        res["equipment_Reefer"] = (res["equipment"] == "Reefer").astype(float)
        res["equipment_Flatbed"] = (res["equipment"] == "Flatbed").astype(float)

        # 7. Temporal & Cyclical Calendar Dynamics
        dow = res["date"].dt.dayofweek
        doy = res["date"].dt.dayofyear
        month = res["date"].dt.month
        res["day_of_week"] = dow.astype(float)
        res["day_of_month"] = res["date"].dt.day.astype(float)
        res["month"] = month.astype(float)
        res["quarter"] = res["date"].dt.quarter.astype(float)
        res["is_weekend"] = (dow >= 5).astype(float)
        res["is_month_end"] = res["date"].dt.is_month_end.astype(float)

        # Cyclical transformations
        res["sin_dow"] = np.sin(2.0 * np.pi * dow / 7.0)
        res["cos_dow"] = np.cos(2.0 * np.pi * dow / 7.0)
        res["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
        res["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)
        res["sin_month"] = np.sin(2.0 * np.pi * month / 12.0)
        res["cos_month"] = np.cos(2.0 * np.pi * month / 12.0)

        # 8. Market Interaction Features
        res["dist_x_quote"] = res["distance"] * res["quote_signal"]
        res["dist_x_market"] = res["distance"] * res["market_index"]
        res["quote_over_market"] = res["quote_signal"] / (res["market_index"] + 1e-5)
        res["dist_x_ton_miles"] = res["distance"] * (res["ton_miles"] / 1000.0)

        return res

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Returns the list of final numeric feature column names for model training."""
        return [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "haversine_distance",
            "circuitousness_ratio",
            "bearing_angle",
            "lat_diff",
            "lon_diff",
            "ton_miles",
            "is_heavy_load",
            "is_light_load",
            "weight_is_na",
            "market_index_is_na",
            "equipment_Dry_Van",
            "equipment_Reefer",
            "equipment_Flatbed",
            "day_of_week",
            "day_of_month",
            "month",
            "quarter",
            "is_weekend",
            "is_month_end",
            "sin_dow",
            "cos_dow",
            "sin_doy",
            "cos_doy",
            "sin_month",
            "cos_month",
            "dist_x_quote",
            "dist_x_market",
            "quote_over_market",
        ]
