"""Pre-review triage signals for Treatment docs.

Extraction-quality heuristics captured across
docs/data_quality_production_v4_model.md §1–§11.  Each function is
a cheap, pre-bootstrap check computed on the raw
``description`` (+ optionally ``diagnosis``) text.  Consumed by
``bin/triage_treatments`` to build the per-treatment review-
priority CSV.

None of these signals are strict — they're triage heuristics.
Combining several catches more issues than any one alone.
"""

import re
from typing import Any, Dict


# Numbered dichotomous-key couplet — matches lines that start
# with "15." / "3a." / "12b) " etc. followed by anatomical prose.
# Detects §8 (key-body content in description).
_COUPLET_LINE_RE = re.compile(
    r'^\s*\d+[a-z]?[.)]\s+[A-Z]', re.MULTILINE,
)

# "sp. nov." / "spec. nov." / "nov. sp." variants.  Multiple
# occurrences in one description signal a multi-species merge
# (§6).  Handles OCR variants like "spec. llOU." partially by
# matching common spelling.
_SP_NOV_RE = re.compile(
    r'\b(?:sp|spec)\.?\s*(?:et\s+)?nov\b',
    re.IGNORECASE,
)

# Latin morphology suffixes.  Rough heuristic — detects a
# paragraph as Latin-heavy when a large fraction of tokens end
# in one of these.  Robust to OCR noise since inflection
# endings survive typos better than binomials.  §6 detection
# via Latin↔English alternation (taxon_572d470e).
_LATIN_SUFFIXES = (
    'us', 'a', 'um', 'is', 'ae', 'orum', 'arum', 'ibus',
    'atis', 'osis', 'osus', 'osa', 'osum', 'alis', 'ale',
    'ata', 'atum', 'ans', 'ens', 'ata', 'ini', 'ata',
    'ceae',
)
# Include common section headers observed in taxonomic prose.
# Not all can be caught with a suffix regex; a curated set of
# Latin vocabulary words that appear regularly in Latin
# diagnoses.
_LATIN_VOCAB = frozenset({
    'apothecia', 'ascomata', 'ascospora', 'ascosporae',
    'ascosporum', 'asci', 'basidia', 'basidiomata', 'conidiomata',
    'conidia', 'excipulum', 'hyphae', 'lamellae', 'paraphyses',
    'peridium', 'perithecia', 'pileus', 'pilei',
    'receptaculum', 'sessilia', 'stipes', 'stipite',
    'stromata', 'sporae', 'thallus',
    'globosa', 'globosum', 'globosis', 'ovoidea', 'ovoidum',
    'ovoideus', 'brevis', 'brevi', 'longa', 'longi',
    'hyalina', 'hyalinum', 'hyalinus', 'ecoloratus', 'aurea',
    'aureus', 'aureum',
})
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


# Diagnosis / Description header punctuation.  Sources use any of
# `:`, `-`, `–` (en-dash), `—` (em-dash) between the label and the
# body — all count as headers.  taxon_8f93bded's `Diagnosis —`
# opener slipped past the earlier literal-colon regex.
_DIAG_HEADER_RE = re.compile(r'\bDiagnosis\s*[-–—:]')
_DESC_HEADER_RE = re.compile(r'\bDescription\s*[-–—:]')


def count_diagnosis_headers(text: str) -> int:
    """Count `Diagnosis:` / `Diagnosis —` / `Diagnosis –` /
    `Diagnosis-` header occurrences.

    Two or more in one description is a strong multi-species
    signal (each species has its own Diagnosis section).  §6.
    """
    if not text:
        return 0
    return len(_DIAG_HEADER_RE.findall(text))


def count_description_headers(text: str) -> int:
    """Count `Description:` / `Description —` / `Description –` /
    `Description-` header occurrences.

    Similar to count_diagnosis_headers but weaker (some real
    descriptions may include the word 'Description' in prose).
    Use in combination with count_diagnosis_headers.
    """
    if not text:
        return 0
    return len(_DESC_HEADER_RE.findall(text))


def mid_body_description_header(text: str) -> bool:
    """True if a `Description:` header appears at offset > 0
    inside the raw description field WITHOUT a preceding
    Diagnosis header — a species-boundary signal (§6 refinement
    from taxon_a21a83f4).

    Rationale: the description field IS the description.  The
    only legitimate offset for a `Description:` header inside
    it is 0 (if the field carries its own header at all).  A
    mid-body `Description:` at offset > 0 signals a second
    species — UNLESS a `Diagnosis` block precedes it, in which
    case the `Description:` is a section boundary within one
    species (Latin/English Diagnosis followed by the main
    English Description; taxon_8f93bded-shape structure).
    """
    if not text:
        return False
    match = _DESC_HEADER_RE.search(text)
    if match is None or match.start() == 0:
        return False
    preceding = text[:match.start()]
    if _DIAG_HEADER_RE.search(preceding):
        return False
    return True


def count_sp_nov(text: str) -> int:
    """Count `sp. nov.` / `spec. nov.` / `nov. sp.` variants.

    Two or more signals a merge (§6) — a single-species
    treatment has one sp. nov. mention at most.
    """
    if not text:
        return 0
    return len(_SP_NOV_RE.findall(text))


def count_key_couplets(text: str) -> int:
    """Count lines that look like dichotomous-key couplets.

    ``^\\d+[a-z]?[.)]\\s+[A-Z]`` at line start signals a numbered
    key couplet like `15. Basal bulb ...` or `3a) Pileus ...`.
    Two or more in the description signals key-body content
    leakage (§8).
    """
    if not text:
        return 0
    return len(_COUPLET_LINE_RE.findall(text))


def desc_starts_mid_sentence(text: str) -> bool:
    """True if the description opens with punctuation or a
    lowercase letter — signals §10 (clipped-at-head extraction).

    Real descriptions open with a capital letter (section
    header like `Pileus`) or a Latin word (also capitalized).
    Openings like `; perithecia ...` (taxon_acd88732) or
    `, hyaline ...` are telltale extraction failures.
    """
    if not text:
        return False
    # Strip leading whitespace but preserve punctuation.
    stripped = text.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    if first in ';,.:':
        return True
    if first.islower():
        return True
    return False


def _latin_ratio(text: str) -> float:
    """Fraction of tokens that look Latin.  Internal helper for
    ``latin_block_count``; 3+-letter tokens matching one of the
    Latin suffixes OR appearing in a curated vocabulary count as
    Latin.  Non-tokens (whitespace, punctuation, numbers) are
    ignored in the denominator.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 3]
    if not tokens:
        return 0.0
    latin = 0
    for tok in tokens:
        if tok in _LATIN_VOCAB:
            latin += 1
            continue
        # Suffix check — 3+ char suffix takes precedence.  Use the
        # longest matching suffix wins by iterating longer ones
        # first (roughly ordered by length above).
        for suf in _LATIN_SUFFIXES:
            if tok.endswith(suf) and len(tok) > len(suf) + 1:
                latin += 1
                break
    return latin / len(tokens)


def latin_block_count(text: str, threshold: float = 0.35) -> int:
    """Count distinct 'Latin blocks' in text.

    Paragraphs (delimited by blank lines) are individually scored
    by ``_latin_ratio``; contiguous paragraphs above ``threshold``
    count as one block.  Real single-species treatments have 0 or
    1 Latin blocks (English throughout, or one Latin diagnosis
    then English).  Two or more blocks signal a multi-species
    merge (§6, taxon_572d470e case).

    threshold=0.35 chosen roughly — Latin taxonomic diagnoses
    typically score ~0.5-0.7 on this measure, English descriptions
    ~0.1-0.2.  A block below 0.35 is definitively English; above,
    likely Latin.
    """
    if not text:
        return 0
    paragraphs = re.split(r'\n\s*\n', text)
    block_count = 0
    in_latin = False
    for p in paragraphs:
        if not p.strip():
            continue
        if _latin_ratio(p) >= threshold:
            if not in_latin:
                block_count += 1
                in_latin = True
        else:
            in_latin = False
    return block_count


# ---------------------------------------------------------------------------
# Composed helpers over a full Treatment doc
# ---------------------------------------------------------------------------


def treatment_signals(treatment: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all triage signals for a Treatment doc, return
    as a flat dict suitable for CSV columns.

    The returned dict is intentionally verbose — every derived
    value is a column so operators can eyeball the raw signals,
    not just the summary verdict.
    """
    desc = treatment.get('description') or ''
    diag = treatment.get('diagnosis') or ''
    return {
        'desc_length': len(desc),
        'diag_length': len(diag),
        'n_diagnosis_headers': count_diagnosis_headers(desc),
        'n_description_headers': count_description_headers(desc),
        'n_sp_nov': count_sp_nov(desc),
        'n_key_couplets': count_key_couplets(desc),
        'desc_starts_mid_sentence':
            desc_starts_mid_sentence(desc),
        'latin_block_count': latin_block_count(desc),
        'mid_body_description_header':
            mid_body_description_header(desc),
        'synthetic_nomenclature':
            bool(treatment.get('synthetic_nomenclature')),
    }


def predicted_issues(
    signals: Dict[str, Any],
    merge_metric: int,
    merge_threshold: int = 10,
) -> str:
    """Concatenate triggered-issue flags into a compact string
    for the CSV.  Empty string = no flags = probably clean.

    Order matches the memo's §-numbering roughly.  Multiple
    flags can fire; the CSV consumer sorts by priority.
    """
    flags = []
    if signals.get('synthetic_nomenclature'):
        flags.append('§2:synth_nomen')
    if signals.get('desc_starts_mid_sentence'):
        flags.append('§10:mid_sentence')
    if signals.get('desc_length', 0) < 500 and signals.get(
        'n_key_couplets', 0,
    ) >= 1:
        flags.append('§8:key_content_short')
    elif signals.get('n_key_couplets', 0) >= 2:
        flags.append('§8:key_couplets')
    if signals.get('n_diagnosis_headers', 0) >= 2:
        flags.append('§6:multi_diagnosis')
    if signals.get('n_description_headers', 0) >= 2:
        flags.append('§6:multi_description')
    # §6 refinement (taxon_a21a83f4): a `Description:` header at
    # offset > 0 without a preceding Diagnosis header marks a
    # species boundary even when count == 1.  Independent of the
    # multi_description flag — both can fire.
    if signals.get('mid_body_description_header'):
        flags.append('§6:mid_body_desc')
    if signals.get('n_sp_nov', 0) >= 2:
        flags.append('§6:multi_sp_nov')
    if signals.get('latin_block_count', 0) >= 2:
        flags.append('§6:latin_alt')
    if merge_metric >= merge_threshold:
        flags.append(f'§6:merge_metric={merge_metric}')
    return '|'.join(flags)


__all__ = (
    'count_diagnosis_headers',
    'count_description_headers',
    'count_sp_nov',
    'count_key_couplets',
    'desc_starts_mid_sentence',
    'latin_block_count',
    'mid_body_description_header',
    'treatment_signals',
    'predicted_issues',
)
