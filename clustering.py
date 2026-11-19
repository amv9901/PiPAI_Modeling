"""Gym member segments: who should get the monthly offer? (k-means)

Run: python clustering.py [k]
Writes outputs/clustering_results.txt and outputs/clustering_elbow.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
CHOSEN_K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CREDIT_COST = 10
SPEND_CUTOFF = 20
LIFT_MARGIN = 25 * 0.60  # extra $25 spend at 60% margin = $15

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


# member_id is just a label, so it is left out; everything else describes the member
FEATS = ["age", "visits_per_month", "avg_visit_minutes", "classes_per_month",
         "months_member", "monthly_extras_spend"]

df = pd.read_csv(HERE / "data" / "gym_members.csv")
log(f"Data: {df.shape[0]} rows x {df.shape[1]} cols; mean extras spend ${df.monthly_extras_spend.mean():.2f}")

# 1. Split 70/30, scale on train only
train, test = train_test_split(df, test_size=0.30, random_state=SEED)
scaler = StandardScaler().fit(train[FEATS])
Ztr, Zte = scaler.transform(train[FEATS]), scaler.transform(test[FEATS])
log(f"Split: {len(train)} train / {len(test)} test; StandardScaler fit on train; features = {FEATS}")

# 2 & 3. k = 1..8, train and test spread
models, tr_sse, te_sse = {}, [], []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Ztr)
    models[k] = km
    tr_sse.append(km.inertia_)
    d = ((Zte[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    te_sse.append(d.min(axis=1).sum())
log("\n k | train spread | test spread | test % drop vs k-1")
for k in range(1, 9):
    drop = "" if k == 1 else f"{(1 - te_sse[k - 1] / te_sse[k - 2]):.1%}"
    log(f" {k} | {tr_sse[k - 1]:12.1f} | {te_sse[k - 1]:11.1f} | {drop}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, sse, title in [(axes[0], tr_sse, f"Training members (n={len(train)})"),
                       (axes[1], te_sse, f"Test members (n={len(test)})")]:
    ax.plot(range(1, 9), sse, "o-")
    ax.axvline(CHOSEN_K, color="gray", ls="--", lw=1)
    ax.set_xlabel("Number of groups (k)")
    ax.set_ylabel("Within-group spread (sum of squared distances)")
    ax.set_title(title)
fig.tight_layout()
fig.savefig(OUT / "clustering_elbow.png", dpi=200)

km = models[CHOSEN_K]
log(f"\nChosen k = {CHOSEN_K}. Test spread k=1 (baseline) = {te_sse[0]:.1f}; k={CHOSEN_K} = {te_sse[CHOSEN_K - 1]:.1f}"
    f" ({1 - te_sse[CHOSEN_K - 1] / te_sse[0]:.1%} lower)")

# 5. Assign everyone, profile segments
df["segment"] = km.predict(scaler.transform(df[FEATS]))
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATS)
profile = df.groupby("segment")[FEATS].mean().round(1)
profile["members"] = df.segment.value_counts().sort_index()
profile["share_spend_lt_20"] = df.groupby("segment").monthly_extras_spend.apply(lambda s: (s < SPEND_CUTOFF).mean()).round(2)
log("\nCentroids (training fit, original units):\n" + centers.round(1).to_string())
log("\nSegment profile, all 330 members:\n" + profile.to_string())

# 4. Credit decision: give the credit to segments where it nets money on average
df["net_if_credit"] = np.where(df.monthly_extras_spend < SPEND_CUTOFF, LIFT_MARGIN - CREDIT_COST, -CREDIT_COST)
seg_net = df.groupby("segment").net_if_credit.mean()
CREDIT_SEGMENTS = set(seg_net[seg_net > 0].index)
log("\nAverage net per member if credited, by segment:\n" + seg_net.round(2).to_string())
log(f"Segments that get the credit: {sorted(CREDIT_SEGMENTS)}")


def assign(member: dict) -> tuple[int, bool]:
    """Return (segment, gets_credit) for one member's features."""
    z = scaler.transform(pd.DataFrame([member])[FEATS])
    seg = int(km.predict(z)[0])
    return seg, seg in CREDIT_SEGMENTS


new = dict(age=30, visits_per_month=4, avg_visit_minutes=42, classes_per_month=1, months_member=5, monthly_extras_spend=10)
seg, credit = assign(new)
log(f"\nNew member -> segment {seg}, gets credit: {credit}")

# Value: member by member
base_net = df.net_if_credit.sum()
v1_net = df.loc[df.segment.isin(CREDIT_SEGMENTS), "net_if_credit"].sum()
log(f"\nMonthly net from the credit, all {len(df)} members:")
log(f"  Baseline (credit to everyone): ${base_net:,.0f}")
log(f"  v1 (credit to segments {sorted(CREDIT_SEGMENTS)} only, {df.segment.isin(CREDIT_SEGMENTS).sum()} members): ${v1_net:,.0f}")
log(f"  Difference: ${v1_net - base_net:,.0f}/month")

(OUT / "clustering_results.txt").write_text("\n".join(lines), encoding="utf-8")
