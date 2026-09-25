# Methodology draft

The production baseline uses deterministic Unicode-aware normalization while retaining original business fields. Candidate generation unions normalized and compact name exact matches, name and address token inverted indices, country/name exact matches, and a compact-name prefix block. Pair features combine name/address string similarity, token Jaccard, exact indicators, country/source indicators, and cross-field similarity interactions.

The classifier is a CPU-compatible histogram gradient booster. Source-1 identifiers, rather than individual pairs, are split for validation. Threshold selection optimizes macro F0.5 with explicit singleton predictions. Candidate recall and validation score: **[TBD AFTER TRAINING]**.
