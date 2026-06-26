"""treatments_to_structured — Phase 1 toolkit for bootstrapping
LLM-annotated, human-reviewed feature/value training data from
Treatment prose.

See docs/schema_constrained_pipeline.md §10 for the phase plan.

This package CONSUMES Treatments (output of skol_classifier v4) and
produces structured annotations for human review and eventual
training/eval data.  Not under skol_classifier/ deliberately —
skol_classifier produces Treatments; this package starts from them.
"""

__version__ = "0.0.1"
