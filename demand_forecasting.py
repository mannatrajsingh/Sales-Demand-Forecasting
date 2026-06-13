"""
demand_forecasting.py
---------------------
End-to-end daily demand forecasting on store-sales data.

Pipeline:
  1. Load data (synthetic Rossmann-style, or swap in real Rossmann train.csv)
  2. Feature engineering: calendar parts, promo/holiday flags, and the key
     temporal signals -> LAG and ROLLING-WINDOW features (computed per store,
     past-only, so there is no leakage)
  3. Time-aware split: train on the past, validate on the most recent weeks
  4. Seasonal-naive baseline: "same weekday last week" (lag-7)
  5. LightGBM model with early stopping
  6. Evaluation: RMSE / MAE / MAPE for BOTH, reported as % improvement over
     the baseline (the number that actually proves the model adds value)
  7. Feature importance + predictions-vs-actual plots
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "/home/claude/store_sales.csv"   # <- point this at Rossmann train.csv to use real data
VALID_WEEKS = 6                              # last 6 weeks held out for validation


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    # Forecasting is only meaningful on open days with real sales.
    df = df[df["Open"] == 1].copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def add_calendar_features(df):
    d = df["Date"].dt
    df["dayofweek"] = d.dayofweek
    df["day"] = d.day
    df["month"] = d.month
    df["year"] = d.year
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = d.is_month_start.astype(int)
    df["is_month_end"] = d.is_month_end.astype(int)
    return df


def add_lag_features(df):
    """
    Lag and rolling features, computed PER STORE and shifted so each row only
    ever sees the PAST. This is the part that makes a tree model able to learn
    temporal structure -- and the part where leakage usually creeps in.
    """
    g = df.groupby("Store")["Sales"]

    # plain lags: sales N days ago
    for lag in (7, 14, 28):
        df[f"lag_{lag}"] = g.shift(lag)

    # rolling stats over a PAST window. shift(1) first so the current day is
    # never included in its own rolling average (that would be leakage).
    shifted = g.shift(1)
    for win in (7, 30):
        df[f"roll_mean_{win}"] = (
            shifted.groupby(df["Store"]).rolling(win).mean().reset_index(level=0, drop=True)
        )
        df[f"roll_std_{win}"] = (
            shifted.groupby(df["Store"]).rolling(win).std().reset_index(level=0, drop=True)
        )
    return df


def build_features(df):
    df = add_calendar_features(df)
    df = add_lag_features(df)

    # StoreType is categorical -> integer codes for LightGBM
    df["StoreType"] = df["StoreType"].astype("category").cat.codes

    # rows without enough history (first 28 days/store) have NaN lags -> drop
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# 3. TIME-AWARE SPLIT  (never shuffle a time series)
# ----------------------------------------------------------------------------
def time_split(df, valid_weeks=VALID_WEEKS):
    cutoff = df["Date"].max() - pd.Timedelta(weeks=valid_weeks)
    train = df[df["Date"] <= cutoff].copy()
    valid = df[df["Date"] > cutoff].copy()
    return train, valid, cutoff


# ----------------------------------------------------------------------------
# 4-6. METRICS, BASELINE, MODEL
# ----------------------------------------------------------------------------
def rmse(y, p):  return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p):   return float(np.mean(np.abs(y - p)))
def mape(y, p):                       # ignore zero-actuals to avoid div-by-0
    m = y > 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


FEATURES = [
    "Promo", "IsHoliday", "StoreType",
    "dayofweek", "day", "month", "year", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end",
    "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_30", "roll_std_30",
]


def run():
    print("Loading data ...")
    df = load_data()
    print(f"  open-day rows: {len(df):,}")

    print("Engineering features ...")
    df = build_features(df)
    print(f"  rows after feature build: {len(df):,}  | features: {len(FEATURES)}")

    train, valid, cutoff = time_split(df)
    print(f"Time-aware split at {cutoff.date()}  ->  "
          f"train {len(train):,} | valid {len(valid):,}")

    Xtr, ytr = train[FEATURES], train["Sales"].to_numpy()
    Xva, yva = valid[FEATURES], valid["Sales"].to_numpy()

    # --- Baseline: seasonal-naive = sales from the same weekday last week -----
    base_pred = valid["lag_7"].to_numpy()

    # --- LightGBM ------------------------------------------------------------
    print("Training LightGBM ...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    model_pred = model.predict(Xva, num_iteration=model.best_iteration_)
    model_pred = np.clip(model_pred, 0, None)

    # --- Evaluation ----------------------------------------------------------
    rows = []
    for name, pred in [("Seasonal-naive baseline", base_pred), ("LightGBM", model_pred)]:
        rows.append({"model": name,
                     "RMSE": rmse(yva, pred),
                     "MAE": mae(yva, pred),
                     "MAPE": mape(yva, pred)})
    res = pd.DataFrame(rows)
    print("\n=== Validation results (last {} weeks) ===".format(VALID_WEEKS))
    print(res.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    b, m = res.iloc[0], res.iloc[1]
    print("\nImprovement over baseline:")
    print(f"  RMSE  {(1 - m.RMSE / b.RMSE) * 100:5.1f}% lower")
    print(f"  MAE   {(1 - m.MAE  / b.MAE ) * 100:5.1f}% lower")
    print(f"  MAPE  {(b.MAPE - m.MAPE):5.1f} pts lower "
          f"({m.MAPE:.1f}% vs {b.MAPE:.1f}%)")

    # --- Plot 1: feature importance -----------------------------------------
    imp = (pd.Series(model.feature_importances_, index=FEATURES)
           .sort_values(ascending=True))
    plt.figure(figsize=(8, 6))
    plt.barh(imp.index, imp.values, color="#4C8BF5")
    plt.title("LightGBM feature importance (gain-split count)")
    plt.tight_layout()
    plt.savefig("/home/claude/feature_importance.png", dpi=130)
    plt.close()

    # --- Plot 2: predictions vs actual for one sample store -----------------
    sid = valid["Store"].value_counts().index[0]
    sv = valid[valid["Store"] == sid].sort_values("Date")
    svp = np.clip(model.predict(sv[FEATURES], num_iteration=model.best_iteration_), 0, None)
    plt.figure(figsize=(11, 4.5))
    plt.plot(sv["Date"], sv["Sales"], label="Actual", marker="o", ms=3)
    plt.plot(sv["Date"], svp, label="LightGBM forecast", marker="x", ms=4)
    plt.plot(sv["Date"], sv["lag_7"], label="Seasonal-naive", ls="--", alpha=0.6)
    plt.title(f"Store {sid}: forecast vs actual (validation window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/forecast_vs_actual.png", dpi=130)
    plt.close()

    print("\nTop 5 drivers:", ", ".join(imp.sort_values(ascending=False).index[:5]))
    print("Saved plots -> feature_importance.png, forecast_vs_actual.png")
    return res


if __name__ == "__main__":
    run()
