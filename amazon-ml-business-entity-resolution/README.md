# Amazon ML Business Entity Resolution

CPU-first two-stage entity-resolution pipeline using only the challenge TSVs. It normalizes data, builds a union of exact/token/address/prefix blocks, trains a source-1-level validation split with `HistGradientBoostingClassifier`, selects a macro F0.5 threshold, and writes a valid submission.

## Setup and commands

```powershell
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/validate_submission.py
```

Data must be under `data/train` and `data/test`. The supplied TSVs belong there. For separate stages use `python scripts/train.py`, `python scripts/predict.py`, and `python scripts/generate_candidates.py --split test`.

Results are written to `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The model is saved as `models/matcher.joblib`. The pipeline never uses external data; it runs on CPU and uses deterministic seed 42 by default.
