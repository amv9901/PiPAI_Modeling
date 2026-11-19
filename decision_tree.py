"""Clinic no-shows: should the clinic overbook this slot? (decision tree)

Run: python decision_tree.py
Writes outputs/decision_tree_results.txt and outputs/decision_tree.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

SEED = 42
MAX_DEPTH = 3
MIN_LEAF = 10
ANNUAL_APPOINTMENTS = 5000  # assumption: ~20 appointments/day x 250 clinic days
VALUE = {"TN": 0, "FP": -60, "FN": -120, "TP": 0}  # positive class = no-show

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


FEATS = ["lead_time_days", "prior_no_shows", "age", "distance_miles",
         "sms_reminder", "new_patient", "appointment_hour"]

df = pd.read_csv(HERE / "data" / "clinic_appointments.csv")
df["y"] = (df.no_show == "no-show").astype(int)
log(f"Data: {df.shape[0]} rows x {df.shape[1] - 1} cols; no-shows {df.y.sum()} ({df.y.mean():.1%})")

# 1. Stratified 70/30 split
train, test = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df.y)
log(f"Split: {len(train)} train ({train.y.mean():.1%} no-show) / {len(test)} test ({test.y.mean():.1%} no-show)")

# 2. Fit v1
tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=SEED)
tree.fit(train[FEATS], train.y)
log(f"\nv1: DecisionTreeClassifier(max_depth={MAX_DEPTH}, min_samples_leaf={MIN_LEAF}), features = {FEATS}")
log(export_text(tree, feature_names=FEATS, show_weights=True))
imp = pd.Series(tree.feature_importances_, index=FEATS).sort_values(ascending=False)
log("Feature importance:\n" + imp.round(3).to_string())

fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(tree, feature_names=FEATS, class_names=["show", "no-show"], filled=True,
          impurity=False, proportion=False, rounded=True, fontsize=9, ax=ax)
fig.tight_layout()
fig.savefig(OUT / "decision_tree.png", dpi=200)


# 3. Evaluate on the test set
def report(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    total = tn * VALUE["TN"] + fp * VALUE["FP"] + fn * VALUE["FN"] + tp * VALUE["TP"]
    n = len(y_true)
    log(f"{name}: TN={tn} FP={fp} FN={fn} TP={tp} | accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f}")
    log(f"   value on {n} test appts = ${total:,.0f}  (${total / n:,.2f} per appt;"
        f" ${total / n * ANNUAL_APPOINTMENTS:,.0f}/yr at {ANNUAL_APPOINTMENTS:,} appts)")
    return total / n


log("\n=== Test appointments ===")
base_pp = report("Baseline (everyone shows)", test.y, np.zeros(len(test), dtype=int))
v1_pp = report("v1 tree", test.y, tree.predict(test[FEATS]))
log(f"v1 training accuracy = {accuracy_score(train.y, tree.predict(train[FEATS])):.3f}"
    f" vs test accuracy = {accuracy_score(test.y, tree.predict(test[FEATS])):.3f}")
log(f"Improvement per appointment = ${v1_pp - base_pp:,.2f}; per year = ${(v1_pp - base_pp) * ANNUAL_APPOINTMENTS:,.0f}")


# 4. Decision rule
def overbook_or_hold(appt: dict) -> str:
    """Overbook if the tree predicts a no-show, otherwise hold the slot."""
    row = pd.DataFrame([appt])[FEATS]
    return "overbook" if tree.predict(row)[0] == 1 else "hold"


def leaf_path(appt: dict) -> str:
    row = pd.DataFrame([appt])[FEATS]
    node_ids = tree.decision_path(row).indices
    t = tree.tree_
    steps = []
    for node in node_ids:
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            p = counts[1] / counts.sum()
            steps.append(f"leaf #{node} ({int(t.n_node_samples[node])} train appts, {p:.0%} no-show)")
        else:
            f, thr = FEATS[t.feature[node]], t.threshold[node]
            steps.append(f"{f} {'<=' if row.iloc[0][f] <= thr else '>'} {thr:.1f}")
    return " -> ".join(steps)


D = dict(lead_time_days=21, prior_no_shows=1, age=27, distance_miles=9.5, sms_reminder=0, new_patient=1, appointment_hour=15)
E = dict(lead_time_days=3, prior_no_shows=0, age=61, distance_miles=2.0, sms_reminder=1, new_patient=0, appointment_hour=9)
log("\n=== Upcoming appointments ===")
for name, appt in [("D", D), ("E", E)]:
    log(f"{name}: {overbook_or_hold(appt).upper()}  | path: {leaf_path(appt)}")

(OUT / "decision_tree_results.txt").write_text("\n".join(lines), encoding="utf-8")
