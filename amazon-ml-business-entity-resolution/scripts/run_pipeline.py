from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from business_entity_resolution.core import train, predict
p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Path("data"));p.add_argument("--model-dir",type=Path,default=Path("models"));p.add_argument("--output-dir",type=Path,default=Path("output"));p.add_argument("--seed",type=int,default=42)
if __name__ == "__main__":
 a=p.parse_args(); print(train(a.data_dir,a.model_dir,a.seed)); predict(a.data_dir,a.model_dir,a.output_dir)
