from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from business_entity_resolution.core import load_split, generate_candidates, candidate_tsv
p=argparse.ArgumentParser(); p.add_argument("--split",choices=["train","test"],default="test"); p.add_argument("--data-dir",type=Path,default=Path("data")); p.add_argument("--output",type=Path,default=Path("output/candidate_pairs.tsv"))
if __name__ == "__main__":
 a=p.parse_args(); s1,v=load_split(a.data_dir,a.split); a.output.parent.mkdir(parents=True,exist_ok=True); candidate_tsv(generate_candidates(s1,v)).to_csv(a.output,sep="\t",index=False)
