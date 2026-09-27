from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from business_entity_resolution.core import train, predict
p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Path("data"));p.add_argument("--model-dir",type=Path,default=Path("models"));p.add_argument("--output-dir",type=Path,default=Path("output"));p.add_argument("--seed",type=int,default=42)
if __name__ == "__main__":
    a = p.parse_args()
    print("=" * 60, flush=True)
    print("Starting Amazon ML Business Entity Resolution Pipeline", flush=True)
    print("=" * 60, flush=True)
    print("[1/3] Loading training dataset & generating candidates...", flush=True)
    metrics = train(a.data_dir, a.model_dir, a.seed)
    print("\n[2/3] Training completed. Validation Metrics:", flush=True)
    print(metrics, flush=True)
    print("\n[3/3] Generating predictions on test dataset...", flush=True)
    predict(a.data_dir, a.model_dir, a.output_dir)
    print("\nPipeline finished successfully! Outputs written to:", flush=True)
    print(f"  - {a.output_dir / 'matching_results.tsv'}", flush=True)
    print(f"  - {a.output_dir / 'candidate_pairs.tsv'}", flush=True)

