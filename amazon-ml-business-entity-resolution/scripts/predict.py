from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from business_entity_resolution.core import predict
p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Path("data")); p.add_argument("--model-dir",type=Path,default=Path("models")); p.add_argument("--output-dir",type=Path,default=Path("output"))
if __name__ == "__main__": predict(**vars(p.parse_args()))
