import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from business_entity_resolution.core import enrich, generate_candidates, score_sets

def test_blocking_and_metric_handle_singletons():
    s1 = enrich(pd.DataFrame([["S1-1", "Acme & Co.", "10 Main St", "US"]], columns=["entity_id", "business_name", "business_address", "country"]))
    vendor = enrich(pd.DataFrame([["S2-1", "Acme and Co", "10 Main Street", "US"]], columns=["entity_id", "business_name", "business_address", "country"]))
    candidates = generate_candidates(s1, vendor)
    assert candidates.candidate_ids.iloc[0] == ["S2-1"]
    assert score_sets({"S1-1": set()}, {"S1-1": set()})["macro_f0_5"] == 1.0
