"""CPU-first, deterministic business entity-resolution pipeline."""
from __future__ import annotations
import json, re, unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

REQUIRED = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

def _norm(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).lower().replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)

def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep originals and add deterministic normalized name/address columns."""
    result = frame.copy()
    for field in ("business_name", "business_address"):
        result[f"{field}_original"] = result[field].fillna("").astype(str)
        result[f"{field}_normalized"] = result[f"{field}_original"].map(_norm)
        result[f"{field}_compact"] = result[f"{field}_normalized"].map(_compact)
        result[f"{field}_tokens"] = result[f"{field}_normalized"].map(lambda x: tuple(x.split()))
    result["country_normalized"] = result["country"].fillna("").astype(str).map(_norm)
    return result

def load_source(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED) - set(frame.columns)
    extra = set(frame.columns) - set(REQUIRED)
    if missing or extra:
        raise ValueError(f"{path}: missing={sorted(missing)}, unexpected={sorted(extra)}")
    if frame.entity_id.duplicated().any():
        raise ValueError(f"{path}: duplicate entity_id values")
    if not frame.entity_id.str.startswith(prefix).all():
        raise ValueError(f"{path}: expected every ID to start with {prefix}")
    return enrich(frame)

def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = data_dir / split
    s1 = load_source(root / f"{split}_source1.tsv", "S1-")
    s2 = load_source(root / f"{split}_source2.tsv", "S2-")
    s3 = load_source(root / f"{split}_source3.tsv", "S3-")
    return s1, pd.concat([s2, s3], ignore_index=True)

def load_truth(path: Path, source1_ids: set[str]) -> dict[str, set[str]]:
    truth = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(truth.columns) != GROUND_TRUTH_COLUMNS or truth.source1_entity_id.duplicated().any():
        raise ValueError("invalid ground-truth schema or duplicate S1 IDs")
    if not set(truth.source1_entity_id).issubset(source1_ids):
        raise ValueError("ground truth contains unknown S1 IDs")
    result = {entity: set() for entity in source1_ids}
    for row in truth.itertuples(index=False):
        matches = set(filter(None, row.matched_entity_ids.split(",")))
        if any(not item.startswith(("S2-", "S3-")) for item in matches):
            raise ValueError(f"invalid matched ID for {row.source1_entity_id}")
        result[row.source1_entity_id] = matches
    return result

def _index(records: pd.DataFrame, column: str, tokenized: bool = False) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for row in records[["entity_id", column]].itertuples(index=False):
        values = row[1] if tokenized else (row[1],)
        for value in values:
            if value and len(value) > 1:
                index[value].add(row.entity_id)
    return index

def generate_candidates(s1: pd.DataFrame, vendors: pd.DataFrame, max_per_record: int = 500) -> pd.DataFrame:
    """Union exact, compact, token, address, country-aware and n-gram blocks."""
    fields = [("business_name_normalized", False), ("business_name_compact", False),
              ("business_name_tokens", True), ("business_address_tokens", True)]
    indices = [_index(vendors, field, tokenized) for field, tokenized in fields]
    country_name = _index(vendors.assign(key=vendors.country_normalized + "|" + vendors.business_name_normalized), "key")
    prefix_index: dict[str, set[str]] = defaultdict(set)
    for row in vendors[["entity_id", "business_name_compact"]].itertuples(index=False):
        if len(row.business_name_compact) >= 5:
            prefix_index[row.business_name_compact[:5]].add(row.entity_id)
    rows = []
    for item in s1.itertuples(index=False):
        found: set[str] = set()
        for (field, tokenized), index in zip(fields, indices):
            values = getattr(item, field) if tokenized else (getattr(item, field),)
            for value in values:
                found.update(index.get(value, ()))
        found.update(country_name.get(f"{item.country_normalized}|{item.business_name_normalized}", ()))
        # Prefix n-gram blocking catches common OCR/truncation changes cheaply.
        name = item.business_name_compact
        if len(name) >= 5:
            found.update(prefix_index.get(name[:5], ()))
        rows.append((item.entity_id, sorted(found)[:max_per_record]))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_ids"])

def candidate_tsv(candidates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"source1_entity_id": candidates.source1_entity_id,
                         "candidate_entity_ids": candidates.candidate_ids.map(lambda x: ",".join(x))})

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a or b else 1.0

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0

def build_features(s1: pd.DataFrame, vendors: pd.DataFrame, candidates: pd.DataFrame,
                   truth: dict[str, set[str]] | None = None) -> tuple[pd.DataFrame, list[str]]:
    left = s1.set_index("entity_id"); right = vendors.set_index("entity_id"); rows = []
    for entry in candidates.itertuples(index=False):
        source = left.loc[entry.source1_entity_id]
        for candidate_id in entry.candidate_ids:
            target = right.loc[candidate_id]
            ns = _ratio(source.business_name_normalized, target.business_name_normalized)
            ads = _ratio(source.business_address_normalized, target.business_address_normalized)
            row = {"source1_entity_id": entry.source1_entity_id, "candidate_entity_id": candidate_id,
                   "name_exact": float(source.business_name_normalized == target.business_name_normalized),
                   "name_compact_exact": float(source.business_name_compact == target.business_name_compact),
                   "name_similarity": ns, "name_token_jaccard": _jaccard(source.business_name_tokens, target.business_name_tokens),
                   "address_exact": float(source.business_address_normalized == target.business_address_normalized),
                   "address_similarity": ads, "address_token_jaccard": _jaccard(source.business_address_tokens, target.business_address_tokens),
                   "same_country": float(source.country_normalized == target.country_normalized),
                   "name_address_product": ns * ads, "min_similarity": min(ns, ads),
                   "source_is_s2": float(candidate_id.startswith("S2-")),
                   "name_length_difference": abs(len(source.business_name_normalized)-len(target.business_name_normalized)),
                   "address_length_difference": abs(len(source.business_address_normalized)-len(target.business_address_normalized))}
            if truth is not None: row["label"] = int(candidate_id in truth.get(entry.source1_entity_id, set()))
            rows.append(row)
    result = pd.DataFrame(rows)
    features = [col for col in result.columns if col not in {"source1_entity_id", "candidate_entity_id", "label"}]
    return result, features

def score_sets(truth: dict[str, set[str]], predicted: dict[str, set[str]]) -> dict[str, float]:
    f_scores=[]; precisions=[]; recalls=[]; tp=fp=fn=0
    for entity, actual in truth.items():
        guess=predicted.get(entity,set()); hit=len(actual & guess); false=len(guess-actual); miss=len(actual-guess)
        precision=hit/(hit+false) if hit+false else (1.0 if not actual else 0.0)
        recall=hit/(hit+miss) if hit+miss else 1.0
        f=(1.25*precision*recall/(.25*precision+recall)) if precision+recall else 0.0
        f_scores.append(f); precisions.append(precision); recalls.append(recall); tp+=hit; fp+=false; fn+=miss
    micro_p=tp/(tp+fp) if tp+fp else 1.0; micro_r=tp/(tp+fn) if tp+fn else 1.0
    return {"macro_precision":float(np.mean(precisions)), "macro_recall":float(np.mean(recalls)), "macro_f0_5":float(np.mean(f_scores)),
            "micro_precision":micro_p, "micro_recall":micro_r, "micro_f0_5":1.25*micro_p*micro_r/(.25*micro_p+micro_r) if micro_p+micro_r else 0.0}

def aggregate(scores: pd.DataFrame, s1_ids: Iterable[str], threshold: float) -> dict[str, set[str]]:
    result={entity:set() for entity in s1_ids}
    for row in scores.loc[scores.probability >= threshold].itertuples(index=False): result[row.source1_entity_id].add(row.candidate_entity_id)
    return result

@dataclass
class TrainedMatcher:
    model: object
    feature_names: list[str]
    threshold: float

def train(data_dir: Path, model_dir: Path, seed: int = 42) -> dict:
    s1, vendors = load_split(data_dir, "train"); truth = load_truth(data_dir / "train" / "train_ground_truth.tsv", set(s1.entity_id))
    candidates = generate_candidates(s1, vendors); pairs, feature_names = build_features(s1, vendors, candidates, truth)
    if pairs.empty or pairs.label.nunique() < 2: raise RuntimeError("blocking produced insufficient labeled candidate pairs")
    ids = s1.entity_id.to_numpy(); train_ids, val_ids = train_test_split(ids, test_size=.2, random_state=seed)
    classifier=HistGradientBoostingClassifier(max_iter=250, learning_rate=.08, max_leaf_nodes=31, l2_regularization=.5, random_state=seed)
    classifier.fit(pairs[pairs.source1_entity_id.isin(train_ids)][feature_names], pairs[pairs.source1_entity_id.isin(train_ids)].label)
    validation=pairs[pairs.source1_entity_id.isin(val_ids)].copy(); validation["probability"]=classifier.predict_proba(validation[feature_names])[:,1]
    best_threshold=.5; best_score=-1.; validation_truth={key:truth[key] for key in val_ids}
    for threshold in np.arange(.50,.951,.05):
        score=score_sets(validation_truth, aggregate(validation, val_ids, float(threshold)))["macro_f0_5"]
        if score > best_score: best_score, best_threshold=score, float(threshold)
    classifier.fit(pairs[feature_names], pairs.label); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(TrainedMatcher(classifier, feature_names, best_threshold), model_dir / "matcher.joblib")
    metrics=score_sets(validation_truth, aggregate(validation,val_ids,best_threshold)); metrics.update({"threshold":best_threshold,"candidate_count":len(pairs),"candidate_recall":float(sum(len(truth[x]&set(candidates.loc[candidates.source1_entity_id.eq(x),'candidate_ids'].iloc[0])) for x in truth)/max(1,sum(map(len,truth.values()))))})
    return metrics

def predict(data_dir: Path, model_dir: Path, output_dir: Path) -> None:
    matcher: TrainedMatcher=joblib.load(model_dir / "matcher.joblib"); s1, vendors=load_split(data_dir,"test")
    candidates=generate_candidates(s1,vendors); pairs, _=build_features(s1,vendors,candidates)
    if not pairs.empty: pairs["probability"]=matcher.model.predict_proba(pairs[matcher.feature_names])[:,1]
    predictions=aggregate(pairs,s1.entity_id,matcher.threshold) if not pairs.empty else {x:set() for x in s1.entity_id}
    output_dir.mkdir(parents=True,exist_ok=True); candidate_tsv(candidates).to_csv(output_dir/"candidate_pairs.tsv",sep="\t",index=False)
    pd.DataFrame({"source1_entity_id":s1.entity_id,"matched_entity_ids":[",".join(sorted(predictions[x])) for x in s1.entity_id]}).to_csv(output_dir/"matching_results.tsv",sep="\t",index=False)
