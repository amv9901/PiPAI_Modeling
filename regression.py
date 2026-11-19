"""Used car prices: should the dealer buy this car? (linear regression)

Run: python regression.py
Writes outputs/regression_results.txt and outputs/regression_pred_vs_actual.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

SEED = 42
PREP_COST = 1500
HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def prep(df):
    X = df[["Mileage_K", "Age_Years", "Horsepower", "Num_Owners"]].copy()
    X["Accident_yes"] = (df["Accident"].str.lower() == "yes").astype(int)
    return X


cars = pd.read_csv(HERE / "data" / "used_car_sales.csv")
lots = pd.read_csv(HERE / "data" / "auction_lots.csv")
log(f"Data: used_car_sales.csv {cars.shape[0]} rows x {cars.shape[1]} cols; auction_lots.csv {lots.shape[0]} rows")
log(f"Mean sale price (all 240): ${cars.Sale_Price.mean():,.0f}")

# 1. Split 70/30
train, test = train_test_split(cars, test_size=0.30, random_state=SEED)
log(f"Split: {len(train)} train / {len(test)} test (random_state={SEED})")

# 2. Fit full model
ALL = ["Mileage_K", "Age_Years", "Horsepower", "Num_Owners", "Accident_yes"]
Xtr, Xte = prep(train), prep(test)
full = sm.OLS(train.Sale_Price, sm.add_constant(Xtr[ALL])).fit()
log("\n=== Full model (all 5 features), training fit ===")
log(full.summary().as_text())

# Drop features whose p-value > 0.05 and refit
drop = [f for f in ALL if full.pvalues[f] > 0.05]
keep = [f for f in ALL if f not in drop]
log(f"\nFeatures with p > 0.05 (candidates to drop): {drop or 'none'}")
reduced = sm.OLS(train.Sale_Price, sm.add_constant(Xtr[keep])).fit() if drop else full
if drop:
    log("\n=== Reduced model, training fit ===")
    log(reduced.summary().as_text())
log(f"\nAdj R2 full = {full.rsquared_adj:.4f}; adj R2 reduced = {reduced.rsquared_adj:.4f}")

# v1 = reduced if it does at least as well on adjusted R2, else full
v1, FEATS = (reduced, keep) if reduced.rsquared_adj >= full.rsquared_adj - 1e-9 else (full, ALL)
log(f"v1 features: {FEATS}")
log("v1 coefficients:")
for k, v in v1.params.items():
    log(f"  {k:>12}: {v:>10,.1f}   p={v1.pvalues[k]:.4g}")
log(f"v1 training R2 = {v1.rsquared:.4f}, adj R2 = {v1.rsquared_adj:.4f}")

# 3. Evaluate on test cars
def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


baseline_price = train.Sale_Price.mean()
y = test.Sale_Price.values
p_base = np.full_like(y, baseline_price, dtype=float)
p_v1 = v1.predict(sm.add_constant(Xte[FEATS], has_constant="add")).values
log("\n=== Test set (cars the model never saw) ===")
log(f"Baseline (train mean ${baseline_price:,.0f}): R2 = {r2(y, p_base):.4f}, RMSE = ${rmse(y, p_base):,.0f}")
log(f"v1:                                R2 = {r2(y, p_v1):.4f}, RMSE = ${rmse(y, p_v1):,.0f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y, p_v1, label="v1 (regression)", alpha=0.8)
ax.scatter(y, p_base, label="baseline (average)", alpha=0.6, marker="x")
lim = [min(y.min(), p_v1.min()) - 1000, max(y.max(), p_v1.max()) + 1000]
ax.plot(lim, lim, "k--", lw=1, label="perfect prediction")
ax.set_xlabel("Actual sale price ($)")
ax.set_ylabel("Predicted sale price ($)")
ax.set_title("Test cars: predicted vs actual sale price")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "regression_pred_vs_actual.png", dpi=200)


# 4. Decision rule
def predict_price(car: dict) -> float:
    row = pd.DataFrame([car])
    row = prep(row)[FEATS]
    return float(v1.predict(sm.add_constant(row, has_constant="add")).iloc[0])


def decide(car: dict, auction_price: float) -> str:
    """Return 'buy' if auction price + prep cost is below the predicted sale price."""
    return "buy" if auction_price + PREP_COST < predict_price(car) else "pass"


def decide_baseline(auction_price: float) -> str:
    return "buy" if auction_price + PREP_COST < baseline_price else "pass"


lots["pred_price"] = [predict_price(r) for r in lots.to_dict("records")]
lots["v1_decision"] = [decide(r, r["Auction_Price"]) for r in lots.to_dict("records")]
lots["base_decision"] = [decide_baseline(a) for a in lots.Auction_Price]
lots["profit_if_bought"] = lots.Later_Sale_Price - lots.Auction_Price - PREP_COST

log("\n=== Auction lots (40 cars) ===")
for name, col in [("Baseline", "base_decision"), ("v1", "v1_decision")]:
    b = lots[lots[col] == "buy"]
    log(f"{name}: buys {len(b)} cars, {int((b.profit_if_bought < 0).sum())} lose money, total profit ${b.profit_if_bought.sum():,.0f}")
log("\nPer-lot detail:")
log(lots[["Lot", "Auction_Price", "pred_price", "Later_Sale_Price", "profit_if_bought", "base_decision", "v1_decision"]]
    .round(0).to_string(index=False))

(OUT / "regression_results.txt").write_text("\n".join(lines), encoding="utf-8")
