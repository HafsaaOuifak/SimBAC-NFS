# SIMBAC-NFS

An interpretable neuro-fuzzy regression model. It builds a large pool of FCM-TSK base learners using both bagging and boosting, then compresses the pool into a small set of fuzzy rules that you can actually read.

---

## How it works

**Phase 1 — Build a hybrid pool**

Run T boosting rounds. In each round, train M bootstrap-resampled FCM-TSK models on the current residuals (bagging within round, boosting across rounds). The pool has T×M base learners.

**Phase 2 — Compress with SIMBA**

Project all rules into a common space, weight each rule by model performance and rule support, cluster similar rules together, and refit Ridge regression on the compressed rule set. What comes out is one small FCM-TSK model you can inspect as IF-THEN statements.

---

## Install

```bash
git clone https://github.com/HafsaaOuifak/SimBAC-NFS.git
cd SimBAC-NFS
pip install -r requirements.txt
```

Datasets download automatically from UCI on first run — nothing is stored in the repo.

---

## Usage

**Command line**

```bash
# single dataset (fast)
python main.py --dataset yacht

# all 9 datasets
python main.py --dataset all

# custom params
python main.py --dataset concrete --tau 0.95 --T 5 --M 3 --nc 20
```

Available datasets: `nasa`, `concrete`, `energy_efficiency`, `ccpp`, `airfoil`, `yacht`, `gas_turbine`, `grid_stability`, `parkinsons`

**Jupyter notebook**

```bash
jupyter lab demo.ipynb
```

Runs the full pipeline on Yacht Hydrodynamics — no local data needed. Shows performance metrics, the fuzzy rule base, and membership function plots.

**Python API**

```python
from src.models.bagging_ensemble import BaggingFCMTSK
from src.models.compression import GradNFSCompressor
import numpy as np

# build pool
T, M, LR = 5, 3, 0.3
y_res, pool = y_train.copy(), []
for t in range(T):
    bag = BaggingFCMTSK(n_estimators=M, n_rules=15, min_rules=3, random_state=42+t)
    bag.fit(X_train, y_res)
    y_res -= LR * np.mean([e.predict(X_train) for e in bag.estimators_], axis=0)
    pool.extend(bag.estimators_)

# compress
model = GradNFSCompressor(tau=0.95, refit_consequents=True).compress(pool, X_train, y_train)

# predict and inspect rules
y_pred = model.predict(X_test)
for i, rule in enumerate(model.get_linguistic_labels(feature_names)):
    print(f"Rule {i+1}: IF " + " AND ".join(f"{f} is {v}" for f, v in rule.items()))
```

---

## Key parameters

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `T` | 5 | boosting rounds |
| `M` | 3 | bags per round |
| `nc` | auto (inner CV) | FCM rule budget per base learner |
| `tau` | 0.95 | how aggressively to merge rules — higher keeps more |
| `LR` | 0.3 | boosting learning rate |

---

## Project structure

```
├── main.py                  CLI runner
├── demo.ipynb               notebook demo
├── src/
│   ├── models/
│   │   ├── fcm_tsk.py       FCM-TSK base learner
│   │   ├── bagging_ensemble.py
│   │   ├── compression.py   SIMBA compression (core)
│   │   └── similarity.py    rule similarity metrics
│   └── datasets/
│       ├── uci_loader.py    UCI downloader
│       └── nasa_battery.py  NASA battery loader
```

---

## Citation

```bibtex
@article{aouifak2026simbac,
  title  = {SIMBAC-NFS: Similarity-Based Antecedent Clustering for Interpretable Neuro-Fuzzy Regression},
  author = {Aouifak, Hafsa and others},
  year   = {2026}
}
```
