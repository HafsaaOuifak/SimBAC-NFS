# SIMBAC-NFS

An interpretable neuro-fuzzy regression model. It builds a large pool of FCM-TSK base learners using both bagging and boosting, then compresses the pool into a small set of fuzzy rules that you can actually read.

---

## How it works

**Phase 1 — Build a hybrid pool**

Run T boosting rounds. In each round, train M bootstrap-resampled FCM-TSK models on the current residuals (bagging within round, boosting across rounds). The pool has T×M base learners.

**Phase 2 — Compress with SIMBA**

Project all rules into a common space, weight each rule by model performance and rule support, cluster similar rules together using antecedent similarity, and refit Ridge regression on the compressed rule set. What comes out is one small FCM-TSK model you can inspect as IF-THEN statements.

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
# single dataset
python main.py --dataset yacht

# all datasets
python main.py --dataset all

# tune it yourself
python main.py --dataset concrete --tau 0.95 --T 5 --M 3 --nc 20 --lr 0.3 --ci 0.99
```

Available datasets: `concrete`, `energy_efficiency`, `ccpp`, `airfoil`, `yacht`, `gas_turbine`, `grid_stability`, `parkinsons`

**Jupyter notebook**

```bash
jupyter lab demo.ipynb
```

Runs the full pipeline on Yacht Hydrodynamics (the simplest dataset — 6 features, 308 samples). No local data needed. Shows performance metrics, the fuzzy rule base, and membership function plots.

**Python API**

```python
from src.models.pool import FCMTSKPool
from src.models.compression import GradNFSCompressor
from src.datasets.data_loader import load_dataset
import numpy as np

X, y, feature_names = load_dataset("yacht")

# build pool (T rounds × M models per round)
T, M, LR = 5, 3, 0.3
y_res, pool = y.copy(), []
for t in range(T):
    round_models = FCMTSKPool(n_estimators=M, n_rules=15, min_rules=3, random_state=42+t)
    round_models.fit(X, y_res)
    y_res -= LR * np.mean([e.predict(X) for e in round_models.estimators_], axis=0)
    pool.extend(round_models.estimators_)

# compress
model = GradNFSCompressor(tau=0.95, refit_consequents=True).compress(pool, X, y)

# read the rules
for i, rule in enumerate(model.get_linguistic_labels(feature_names)):
    print(f"Rule {i+1}: IF " + " AND ".join(f"{f} is {v}" for f, v in rule.items()))
```

---

## Project structure

```
├── main.py                  CLI runner
├── demo.ipynb               notebook demo (Yacht Hydrodynamics)
├── src/
│   ├── models/
│   │   ├── fcm_tsk.py       FCM-TSK base learner
│   │   ├── pool.py          pool construction (rounds × models)
│   │   ├── compression.py   SIMBA compression (core)
│   │   └── similarity.py    rule similarity metrics
│   └── datasets/
│       └── data_loader.py   loads any dataset from UCI
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
