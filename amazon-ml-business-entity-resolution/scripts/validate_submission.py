from pathlib import Path
import subprocess, sys
root=Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable, str(root/'utils'/'validate_submission.py'), '--matching', str(root/'output'/'matching_results.tsv'), '--candidate', str(root/'output'/'candidate_pairs.tsv'), '--test-dir', str(root/'data'/'test')]))
