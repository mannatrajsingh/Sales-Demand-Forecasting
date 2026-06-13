"""
generate_data.py
------------------
Creates a realistic daily store-sales dataset, structured like the Kaggle
Rossmann Store Sales dataset, so the forecasting pipeline can run end-to-end
without needing a Kaggle login.

Signal baked into the data (so the model has something real to learn):
  - store-level base demand (some stores much busier than others)
  - long-term upward trend (store-specific slope)
  - weekly seasonality (weekday vs weekend pattern)
  - yearly seasonality (peaks in Nov-Dec holiday season)
  - promotions (random promo days lift sales)
  - holidays (a fixed set; pre-holiday boost, holiday-day closure)
  - Sunday closures for most stores
  - multiplicative noise

Swap-in for real data: the rest of the pipeline only needs a CSV with columns
[Store, Date, Sales, Promo, StoreType, Open]. Rossmann's train.csv already has
these, so you can point the pipeline at the real file with no other changes.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

NUM_STORES = 1000
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2024-06-30")          # ~2.5 years of daily data

# Fixed "holiday" dates with a closure flag (True = store shut that day)
HOLIDAYS = {
    "2022-01-01": True,  "2022-12-25": True, "2022-12-26": True, "2022-11-24": False,
    "2023-01-01": True,  "2023-12-25": True, "2023-12-26": True, "2023-11-23": False,
    "2024-01-01": True,  "2024-05-27": False,
}
HOLIDAYS = {pd.Timestamp(k): v for k, v in HOLIDAYS.items()}


def generate():
    dates = pd.date_range(START, END, freq="D")
    n_days = len(dates)
    stores = np.arange(1, NUM_STORES + 1)

    # ---- per-store static attributes -------------------------------------
    store_type = RNG.choice(list("abcd"), size=NUM_STORES, p=[0.5, 0.2, 0.2, 0.1])
    base_demand = RNG.lognormal(mean=8.3, sigma=0.45, size=NUM_STORES)   # ~4000 avg
    trend_slope = RNG.normal(0.00018, 0.00010, size=NUM_STORES)          # daily growth
    closes_sunday = RNG.random(NUM_STORES) < 0.85                        # most shut Sun

    # ---- per-day shared signals (same for every store) -------------------
    day_idx = np.arange(n_days)
    dow = dates.dayofweek.to_numpy()                                     # 0=Mon .. 6=Sun
    doy = dates.dayofyear.to_numpy()

    weekday_mult = np.array([1.00, 0.97, 0.96, 0.98, 1.10, 1.25, 0.70])[dow]
    yearly_mult = 1.0 + 0.30 * np.sin(2 * np.pi * (doy - 320) / 365.0)   # peak ~mid-Nov

    is_holiday = np.array([d in HOLIDAYS for d in dates])
    holiday_closed = np.array([HOLIDAYS.get(d, False) for d in dates])
    # day before a holiday gets a shopping bump
    pre_holiday = np.zeros(n_days, dtype=bool)
    pre_holiday[:-1] = is_holiday[1:]
    holiday_mult = np.where(pre_holiday, 1.20, 1.0)

    # ---- build the long table via broadcasting (stores x days) -----------
    # promo: each (store, day) has ~30% chance of a promo, with a sales lift
    promo = (RNG.random((NUM_STORES, n_days)) < 0.30)
    promo_mult = np.where(promo, RNG.uniform(1.15, 1.40, size=(NUM_STORES, n_days)), 1.0)

    noise = RNG.lognormal(mean=0.0, sigma=0.10, size=(NUM_STORES, n_days))

    trend = 1.0 + trend_slope[:, None] * day_idx[None, :]

    sales = (
        base_demand[:, None]
        * trend
        * weekday_mult[None, :]
        * yearly_mult[None, :]
        * holiday_mult[None, :]
        * promo_mult
        * noise
    )

    # ---- openness: closed Sundays (per store) + holiday closures ----------
    is_sunday = (dow == 6)
    closed = (is_sunday[None, :] & closes_sunday[:, None]) | holiday_closed[None, :]
    open_flag = (~closed).astype(int)
    sales = np.where(closed, 0.0, sales)
    promo = np.where(closed, False, promo)              # no promo on closed days

    sales = np.rint(sales).astype(int)

    # ---- flatten to tidy long format -------------------------------------
    df = pd.DataFrame({
        "Store": np.repeat(stores, n_days),
        "Date": np.tile(dates, NUM_STORES),
        "Sales": sales.reshape(-1),
        "Promo": promo.reshape(-1).astype(int),
        "Open": open_flag.reshape(-1),
        "StoreType": np.repeat(store_type, n_days),
        "IsHoliday": np.tile(is_holiday.astype(int), NUM_STORES),
    })
    return df


if __name__ == "__main__":
    df = generate()
    out = "/home/claude/store_sales.csv"
    df.to_csv(out, index=False)
    print(f"rows: {len(df):,}  stores: {df.Store.nunique()}  "
          f"dates: {df.Date.min().date()} -> {df.Date.max().date()}")
    print(df.head())
    print("\nsales summary (open days only):")
    print(df.loc[df.Open == 1, "Sales"].describe().round(1))
    print(f"\nsaved -> {out}")
