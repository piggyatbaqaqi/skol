"""Tests for complexity_score.

Comparative semantics: richer prose → higher score.  We don't
bake in absolute weights here — the calibration step (Phase 1
deliverable 1, see docs/schema_constrained_pipeline.md §10.4)
tunes those by inspection.

The three comparative tests carry ``pytest.mark.xfail`` until
the implementation lands; per CLAUDE.md, HEAD must always have
all tests passing so bisect-style debugging works.  Remove the
xfail markers when complexity_score graduates from
NotImplementedError to a real implementation.
"""

from typing import Any, Dict, Optional

import pytest

from treatments_to_structured.complexity import complexity_score


def _make_treatment(
    description: Optional[str] = None,
    diagnosis: Optional[str] = None,
) -> Dict[str, Any]:
    """Minimal Treatment-doc fixture for complexity_score().

    Matches the production Treatment shape: top-level ``description``
    and ``diagnosis`` are STRING fields holding the prose, either of
    which may be ``None`` (CouchDB null) for treatments whose
    content lives in other fields.

    The companion ``description_spans`` / ``diagnosis_spans`` lists
    (carrying source-plaintext char offsets) are NOT used by the
    complexity scorer — those are consumed by the brat-render module
    later in Phase 1.  We don't construct them here.
    """
    return {
        '_id': 'taxon_test',
        'description': description,
        'diagnosis': diagnosis,
    }


class TestComplexityScore:
    """Behavioral contract for complexity_score.

    Each test asserts a comparative property, not an absolute value.
    The implementation is free to choose its own weighting as long
    as these orderings hold.
    """

    @pytest.mark.xfail(
        reason='complexity_score not yet implemented '
               '(Phase 1 deliverable 1)',
        strict=True,
    )
    def test_returns_float(self) -> None:
        result = complexity_score(_make_treatment(description='hi.'))
        assert isinstance(result, float)

    @pytest.mark.xfail(
        reason='complexity_score not yet implemented '
               '(Phase 1 deliverable 1)',
        strict=True,
    )
    def test_empty_treatment_scores_zero(self) -> None:
        """A treatment with no description / diagnosis prose scores
        0.0 — nothing to annotate, nothing to weight."""
        assert complexity_score(_make_treatment()) == 0.0

    @pytest.mark.xfail(
        reason='complexity_score not yet implemented '
               '(Phase 1 deliverable 1)',
        strict=True,
    )
    def test_null_prose_treated_as_empty_string(self) -> None:
        """Many production Treatments have ``description: null`` or
        ``diagnosis: null`` (the prose lives in ``notes`` or another
        field, which is out of scope for Phase 1).  CouchDB null
        must not crash the scorer — treat it as empty string."""
        all_null = _make_treatment(description=None, diagnosis=None)
        all_empty = _make_treatment(description='', diagnosis='')
        assert complexity_score(all_null) == complexity_score(all_empty)
        assert complexity_score(all_null) == 0.0

    @pytest.mark.xfail(
        reason='complexity_score not yet implemented '
               '(Phase 1 deliverable 1)',
        strict=True,
    )
    def test_one_field_null_other_populated(self) -> None:
        """The Trichoderma sample (docs/schema_constrained_pipeline.md
        §10.1) has ``description: null`` with a short ``diagnosis``.
        The scorer should return whatever the diagnosis alone earns
        — no NaN, no crash on partial nulls."""
        partial = _make_treatment(
            description=None,
            diagnosis='Differs from M. brevicaulis by the absent veil.',
        )
        score = complexity_score(partial)
        assert isinstance(score, float)
        assert score > 0.0

    @pytest.mark.xfail(
        reason='complexity_score not yet implemented '
               '(Phase 1 deliverable 1)',
        strict=True,
    )
    def test_richer_description_scores_higher_than_minimal(self) -> None:
        minimal = _make_treatment(description='A small fungus.')
        rich = _make_treatment(description=(
            'Pileus brown, 3-5 cm wide.  Lamellae cream-colored '
            'when young, ochre at maturity.  Stipe 4 cm long, '
            'cylindrical, smooth.'
        ))
        assert complexity_score(rich) > complexity_score(minimal)

    @pytest.mark.xfail(
        reason='complexity_score not yet implemented '
               '(Phase 1 deliverable 1)',
        strict=True,
    )
    def test_measurement_density_raises_score(self) -> None:
        """Two descriptions of similar word count, one with
        measurements, one without — measurements should win."""
        bland = _make_treatment(description=(
            'Pileus brown.  Lamellae cream.  Stipe long and '
            'cylindrical and smooth and pale.'
        ))
        measured = _make_treatment(description=(
            'Pileus brown 3 cm.  Lamellae cream 5 mm.  Stipe '
            '4 cm long 8 mm wide cylindrical smooth.'
        ))
        assert complexity_score(measured) > complexity_score(bland)

    @pytest.mark.xfail(
        reason='complexity_score not yet implemented '
               '(Phase 1 deliverable 1)',
        strict=True,
    )
    def test_diagnosis_contributes_to_score(self) -> None:
        """Both fields count; a treatment with both Description AND
        Diagnosis prose scores higher than one with Description alone
        (controlling for description content)."""
        desc_only = _make_treatment(
            description='Pileus brown, 3 cm wide.',
        )
        both = _make_treatment(
            description='Pileus brown, 3 cm wide.',
            diagnosis='Differs from M. brevicaulis by the absent veil.',
        )
        assert complexity_score(both) > complexity_score(desc_only)
