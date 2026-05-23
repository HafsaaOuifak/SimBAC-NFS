# SIMBAC-NFS

**SIMBAC-NFS** (SIMilarity-Based Antecedent Clustering for Neuro-Fuzzy Systems) is an interpretable regression model that combines ensemble diversity with a compact, human-readable fuzzy rule base.

It draws on both **bagging** and **boosting**: within each of T sequential residual-learning rounds, M bootstrap-resampled FCM-TSK models are trained (within-round diversity); across rounds, each set focuses on what the previous round left unexplained (across-round diversity). The resulting T×M pool is then compressed by clustering similar fuzzy rules into a small, coherent set — preserving accuracy while delivering a model a domain expert can read and audit.

---

## How It Works

### Phase 1 — Hybrid Pool Construction

```
r ← y                                        # start from original targets
Pool ← {}
for t = 1 … T:
    Train M bootstrap-resampled FCM-TSK models on current residuals r   [bagging]
    r ← r − η · mean_m B_{t,m}(X)                                       [boosting]
    Add M models to Pool
```

Each **FCM-TSK** base learner:
- **IF-part**: Gaussian membership functions whose centers/widths come from Fuzzy C-Means clustering on the (MI-weighted) training data.
- **THEN-part**: Linear (Ridge regression) consequents fitted per cluster.
- Self-pruning: rules below a support threshold are discarded before fitting consequents.

### Phase 2 — SIMBA Compression

```
Project all pool rules into a common normalized space
Weight each rule by: performance weight × support weight

Compute pairwise similarity: S = 0.5·Bhattacharyya + 0.3·Wasserstein + 0.2·centroid
Apply complete-linkage clustering at threshold τ

For each cluster: consolidate center/sigma by importance-weighted average
Keep clusters until cumulative importance ≥ κ

Ridge refit on consolidated antecedents → single compressed FCM-TSK model
```

The output is **one** FCM-TSK model with a dramatically smaller rule base that a domain expert can inspect as IF-THEN statements.

---

## Installation

```bash
git clone https://github.com/<your-username>/simbac-nfs.git
cd simbac-nfs
pip install -r requirements.txt
```

> UCI datasets are downloaded automatically on first use — no data is stored in this repository.

---

## Quick Start

### Command Line

```bash
# Run on Yacht Hydrodynamics (< 1 min)
python main.py --dataset yacht

# Run on Concrete Compressive Strength
python main.py --dataset concrete

# Custom hyperparameters
python main.py --dataset airfoil --nc 20 --tau 0.95 --T 5 --M 3

# All nine benchmark datasets (~2–4 h)
python main.py --dataset all
```

**Available datasets:** `nasa`, `concrete`, `energy_efficiency`, `ccpp`,  
`airfoil`, `yacht`, `gas_turbine`, `grid_stability`, `parkinsons`

Results are saved to `results/csv/<dataset>_main_results.csv`.

### Jupyter Notebook

```bash
jupyter lab demo.ipynb
```

The notebook runs SIMBAC-NFS on Yacht Hydrodynamics end-to-end — no local data files needed — and demonstrates:

- Live data loading from UCI via `ucimlrepo`
- Building the hybrid pool
- SIMBA compression
- RMSE / MAE / R² metrics
- IF-THEN rule base printout
- Membership function visualization
- Interpretability summary (rule count, compression rate)

### Python API

```python
import numpy as np
from src.models.bagging_ensemble import BaggingFCMTSK
from src.models.compression import GradNFSCompressor

# Phase 1: build hybrid pool
T, M, LR = 5, 3, 0.3
y_res = y_train.copy()
pool = []
for t in range(T):
    bag = BaggingFCMTSK(n_estimators=M, n_rules=15, min_rules=3,
                         random_state=42 + t)
    bag.fit(X_train, y_res)
    y_res -= LR * np.mean([e.predict(X_train) for e in bag.estimators_], axis=0)
    pool.extend(bag.estimators_)

# Phase 2: compress
compressor = GradNFSCompressor(
    tau=0.95, similarity_method="combined",
    weight_by_performance=True, cumulative_importance=0.99,
    min_rules=3, refit_consequents=True, tune_refit_alpha=True,
)
model = compressor.compress(pool, X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Inspect rules
for i, rule in enumerate(model.get_linguistic_labels(feature_names)):
    antecedent = " AND ".join(f"{f} is {v}" for f, v in rule.items())
    print(f"Rule {i+1}: IF {antecedent}")
```

---

## Hyperparameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | 5 | Number of boosting rounds |
| `M` | 3 | Bootstrap bags per round |
| `nc` | inner-CV | FCM rule budget per base learner (15/20/25, selected by inner 3-fold CV) |
| `tau` (τ) | 0.95 | Similarity threshold for rule clustering. Higher = stricter, more rules kept |
| `LR` | 0.3 | Boosting learning rate / shrinkage |
| `cumulative_importance` (κ) | 0.99 | Retain clusters covering this fraction of total importance |
| `min_rules` | 3 | Minimum rules to keep after compression |
| `similarity_method` | `combined` | `bhattacharyya`, `wasserstein`, `centroid`, or `combined` |

Per-dataset defaults used in the paper:

| Dataset | τ | T | M | Bootstrap ratio |
|---------|---|---|---|-----------------|
| NASA Battery | 0.73 | 5 | 3 | 1.0 |
| D2–D7, D9 | 0.95 | 5 | 3 | 1.0 |
| Grid Stability | 0.99 | 7 | 5 | 0.03 |

---

## Project Structure

```
simbac-nfs/
├── main.py                      # CLI entry point (runs 5-fold CV experiments)
├── demo.ipynb                   # Interactive demo notebook
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── fcm_tsk.py           # FCM-TSK base learner
│   │   ├── bagging_ensemble.py  # Bootstrap bagging of FCM-TSK models
│   │   ├── compression.py       # SIMBA compression — core contribution
│   │   └── similarity.py        # Antecedent similarity metrics
│   └── datasets/
│       ├── uci_loader.py        # UCI dataset downloader/loader
│       └── nasa_battery.py      # NASA Li-Ion Battery LOBO loader
```

---

## Citation

```bibtex
@article{aouifak2026simbac,
  title   = {SIMBAC-NFS: Similarity-Based Antecedent Clustering
             for Interpretable Neuro-Fuzzy Regression},
  author  = {Aouifak, Hafsa and others},
  journal = {Under review},
  year    = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
