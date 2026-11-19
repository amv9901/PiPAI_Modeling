# PiPAI Modeling Assignments

Code for the three Projects in Programming and AI modeling assignments (Fall 2026). Every number on the slides comes from these scripts.

| Script | Assignment | Model |
|---|---|---|
| `regression.py` | Used car prices: should the dealer buy this car? | Linear regression (statsmodels OLS) |
| `decision_tree.py` | Clinic no-shows: should the clinic overbook this slot? | Classification tree (scikit-learn) |
| `clustering.py` | Gym member segments: who should get the monthly offer? | k-means (scikit-learn) |

Data files are in `data/`, as given with the assignments. All splits are 70/30 with `random_state=42`.

## Run

```bash
pip install -r requirements.txt
python regression.py
python decision_tree.py
python clustering.py
```

Each script prints its results and writes them, along with its charts, to `outputs/`.
