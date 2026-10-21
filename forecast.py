"""Sales forecasting — trend x seasonality model.

Loads data/monthly_sales.csv (36 months), validates the model on a 6-month
holdout (MAPE), then forecasts the next 6 months. Writes charts/ and
forecast_2025_H1.csv.

Run:  python forecast.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
NAVY, GOLD = "#1F3864", "#D9A441"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#ccc"})
df = pd.read_csv(BASE / "data" / "monthly_sales.csv", parse_dates=["Month"])

def fit_forecast(train, steps):
    tt = np.arange(len(train))
    coef = np.polyfit(tt, train["Sales"].values, 1)
    trend = np.polyval(coef, tt)
    idx = pd.Series(train["Sales"].values / trend).groupby(train["Month"].dt.month).mean()
    idx = idx / idx.mean()
    future_t = np.arange(len(train), len(train) + steps)
    future_months = pd.period_range(train["Month"].iloc[-1], periods=steps + 1, freq="M")[1:]
    return np.polyval(coef, future_t) * np.array([idx[m.month] for m in future_months]), future_months.to_timestamp(), idx

train, holdout = df.iloc[:-6], df.iloc[-6:]
pred_hold, hold_months, _ = fit_forecast(train, 6)
mape = float(np.mean(np.abs(holdout["Sales"].values - pred_hold) / holdout["Sales"].values) * 100)
future, future_months, idx = fit_forecast(df, 6)
pd.DataFrame({"Month": future_months.strftime("%Y-%m"), "Forecast": future.round(0),
              "Lower_10pct": (future * 0.9).round(0), "Upper_10pct": (future * 1.1).round(0)}
             ).to_csv(BASE / "forecast_2025_H1.csv", index=False)
print(f"Holdout MAPE: {mape:.1f}%   Forecast total (next 6 months): ${future.sum():,.0f}")

# ---------- charts ----------
(BASE / "charts").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(train["Month"], train["Sales"], color=NAVY, lw=2, label="Actual sales (training)")
ax.plot(holdout["Month"], holdout["Sales"], color=NAVY, lw=2, ls="--", label="Actual sales (holdout)")
ax.plot(hold_months, pred_hold, color=GOLD, lw=2.4, marker="o", label="Model forecast for holdout")
ax.set_title(f"Holdout test — forecast vs actual (MAPE {mape:.1f}%)", fontsize=12, weight="bold", color=NAVY, loc="left")
ax.set_ylabel("Sales ($)"); ax.legend(frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(BASE / "charts" / "holdout_forecast.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(df["Month"], df["Sales"], color=NAVY, lw=2, label="Actual sales 2022–2024")
ax.plot(future_months, future, color=GOLD, lw=2.4, marker="o", label="Forecast Jan–Jun 2025")
ax.fill_between(future_months, future * 0.9, future * 1.1, color=GOLD, alpha=0.18, label="±10% range")
ax.set_title("Sales forecast — first half of 2025", fontsize=12, weight="bold", color=NAVY, loc="left")
ax.set_ylabel("Sales ($)"); ax.legend(frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(BASE / "charts" / "future_forecast.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.bar([str(m) for m in range(1, 13)], idx.values, color=NAVY)
ax.axhline(1.0, color=GOLD, lw=2)
ax.set_title("Seasonal indices (above 1.0 = stronger-than-average month)", fontsize=11.5, weight="bold", color=NAVY, loc="left")
ax.set_xlabel("Month"); ax.set_ylabel("Index"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(BASE / "charts" / "seasonal_indices.png", dpi=150); plt.close(fig)
print("Charts written to charts/.")
