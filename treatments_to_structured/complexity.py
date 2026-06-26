"""Complexity scoring for Treatment docs.

Higher score = richer prose worth annotating during the Phase 1
bootstrap pass.  Used by ``bin/select_for_annotation`` to pick a
sample for the Claude-API annotator.

The score is a comparative signal — there is no canonical
threshold.  Calibration happens by inspecting scored samples and
adjusting the weighted-combo coefficients in this module.  See
docs/schema_constrained_pipeline.md §10.4 deliverable (1).
"""

from typing import Any, Dict


def complexity_score(treatment: Dict[str, Any]) -> float:
    """Score a Treatment doc by prose richness.

    First-cut definition (per the §10 design): weighted combination
    of (a) total prose word count across description + diagnosis,
    (b) feature-keyword hits from a small seed gazetteer (pileus,
    lamellae, stipe, spores, ...), (c) measurement-pattern count
    (``\\d+(\\.\\d+)?\\s*(mm|cm|µm|µ|nm)``).

    Args:
        treatment: A Treatment document with ``description_spans``
            and/or ``diagnosis_spans`` lists, each entry having a
            ``text`` field.  Missing fields treated as empty.

    Returns:
        Non-negative float.  An empty treatment scores 0.0.
        Comparative semantics only — absolute values aren't meaningful.
    """
    raise NotImplementedError(
        "complexity_score is a Phase 1 deliverable.  "
        "See docs/schema_constrained_pipeline.md §10.4 item 1."
    )
