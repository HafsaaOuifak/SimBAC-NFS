# SIMBAC-NFS

Gradient-boosted FCM-TSK pool compressed into a single inspectable fuzzy rule base.

Most ensemble methods trade interpretability for accuracy — you get better predictions but no way to explain them. SIMBAC-NFS takes a different route: it builds a diverse pool of neuro-fuzzy models through gradient boosting, then collapses the whole pool into a handful of IF-THEN rules you can actually read.

On the Combined Cycle Power Plant dataset, for example, ~300 pool rules compress down to 8:

```
         AT        V         AP        RH
Rule 1 [ Low     | High    | Medium  | Low    ]  →  PE ≈ 483 MW
Rule 2 [ High    | Low     | High    | Medium ]  →  PE ≈ 441 MW
Rule 3 [ Medium  | Medium  | Medium  | High   ]  →  PE ≈ 461 MW
Rule 4 [ VeryLow | VeryHigh| Low     | Low    ]  →  PE ≈ 490 MW
Rule 5 [ High    | High    | Low     | VeryLow]  →  PE ≈ 452 MW
Rule 6 [ VeryHigh| Low     | High    | Medium ]  →  PE ≈ 435 MW
Rule 7 [ Low     | Medium  | VeryHigh| High   ]  →  PE ≈ 476 MW
Rule 8 [ Medium  | Low     | Medium  | VeryLow]  →  PE ≈ 444 MW

97.6% of pool rules removed. RMSE = 4.13 MW.
```

Each row is a rule, each column is a sensor reading. A plant engineer can look at Rule 1 and immediately understand: *when it's cool outside with high vacuum pressure, expect maximum output*.

---

## How it works

**Phase 1 — Build a diverse pool**

Run T boosting rounds. Each round trains M FCM-TSK models on the current residuals (what previous rounds got wrong). The pool ends up with T×M models, each capturing different patterns.

**Phase 2 — Compress with SIMBA**

All pool rules are projected into a common feature space. Rules that cover the same region of input space are merged by antecedent similarity clustering (complete linkage, threshold τ). What comes out is one small FCM-TSK model — one rule base, readable as IF-THEN statements.

The compression step uses performance-weighted merging, so rules from better-performing pool members contribute more to the final rule centres.

---

## Install

```bash
git clone https://github.com/HafsaaOuifak/SimBAC-NFS.git
cd SimBAC-NFS
pip install -r requirements.txt
```

Datasets download automatically from UCI on first run — nothing is bundled in the repo.

---

## Usage

**Command line**

```bash
# single dataset
python main.py --dataset ccpp

# all datasets
python main.py --dataset all

# custom hyperparameters
python main.py --dataset concrete --tau 0.95 --T 5 --M 3 --nc 20 --lr 0.3 --ci 0.99
```

Available datasets: `concrete`, `energy_efficiency`, `ccpp`, `airfoil`, `yacht`, `gas_turbine`, `grid_stability`, `parkinsons`

**Notebook**

```bash
jupyter lab demo.ipynb
```

Runs the full pipeline on the Power Plant dataset (9568 samples, 4 features). Walks through 5-fold cross-validation, shows the performance table, interpretability metrics, the compressed rule base, membership function plots, and the rule-centre heatmap.

**Python API**

```python
from src.models.pool import FCMTSKPool
from src.models.compression import SIMBACCompressor
from src.datasets.data_loader import load_dataset
import numpy as np

X, y, feature_names = load_dataset("ccpp")

# gradient-boosted pool: T rounds × M models
T, M, LR = 5, 3, 0.3
y_res, pool = y.copy(), []
for t in range(T):
    bag = FCMTSKPool(n_estimators=M, n_rules=15, min_rules=3, random_state=42+t)
    bag.fit(X, y_res)
    y_res -= LR * np.mean([e.predict(X) for e in bag.estimators_], axis=0)
    pool.extend(bag.estimators_)

# compress the pool into one rule base
model = SIMBACCompressor(tau=0.95, refit_consequents=True).compress(pool, X, y)

# read the rules
for i, rule in enumerate(model.get_linguistic_labels(feature_names)):
    print(f"Rule {i+1}: IF " + " AND ".join(f"{f} is {v}" for f, v in rule.items()))
```

---

## Results

Tested on 9 UCI regression datasets using 5-fold cross-validation (LOBO for NASA Battery).

| Dataset | RMSE | Rules (pool → compressed) | Compression |
|---|---|---|---|
| NASA Battery | 22.0 | 225 → 6 | 97.3% |
| Concrete | 6.6 | 270 → 44 | 83.7% |
| Energy Efficiency | 0.48 | 270 → 69 | 74.3% |
| Power Plant | 4.13 | 315 → 8 | 97.6% |
| Airfoil | 3.22 | 375 → 40 | 89.4% |
| Yacht | 0.63 | 225 → 131 | 41.9% |
| Gas Turbine | 0.0055 | 375 → 25 | 93.4% |
| Grid Stability | 0.0148 | 595 → 584 | 1.8%* |
| Parkinson's | 6.48 | 375 → 45 | 88.1% |

*Grid Stability at τ=0.99 produces highly diverse residual rules that resist merging — a known edge case documented in the paper.

---

## Project structure

```
├── main.py                  CLI runner
├── demo.ipynb               end-to-end notebook (Power Plant)
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── fcm_tsk.py       FCM-TSK base learner
│   │   ├── pool.py          pool construction
│   │   ├── compression.py   SIMBA compression (core algorithm)
│   │   └── similarity.py    rule antecedent similarity metrics
│   └── datasets/
│       └── data_loader.py   UCI dataset loader
```

---

## Citation

```bibtex
@article{ouifak2026simbac,
  title  = {SIMBAC-NFS: Similarity-Based Aggregation and Calibration of Neuro-Fuzzy Systems},
  author = {Ouifak, Hafsaa and Idri, Ali},
  year   = {2026}
}
```
