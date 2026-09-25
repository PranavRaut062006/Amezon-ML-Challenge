from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from business_entity_resolution.core import train
p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Path("data")); p.add_argument("--model-dir",type=Path,default=Path("models")); p.add_argument("--seed",type=int,default=42)
if __name__ == "__main__": print(json.dumps(train(**vars(p.parse_args())),indent=2))
