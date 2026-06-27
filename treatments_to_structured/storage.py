"""CouchDB storage constants and (eventually) helpers for
treatments_to_structured.

Phase 1 deliverable 4.5 declares the single canonical name for the
global golden-features database.  Future deliverables (6 brat
ingestion, 7 promotion to golden) will add the read/write helpers
that reference this constant.
"""


# The per-experiment candidate DB is named via convention in
# bin/manage_experiment.py and bin/llm_annotate_features.py:
# ``skol_exp_<experiment>_02_50_features_candidate``.  It travels
# with the experiment doc (experiment.databases.features_candidate)
# so replication and tooling resolve it through the standard path.
#
# The golden DB, in contrast, is global — not per-experiment, no
# pipeline-order prefix — matching the existing skol_golden_*
# convention (skol_golden_ann_hand, etc.).  Centralized here so
# promotion tooling, eval scorers, and any future readers all share
# one source of truth.
GOLDEN_FEATURES_DB_NAME = 'skol_golden_features'


__all__ = (
    'GOLDEN_FEATURES_DB_NAME',
)
