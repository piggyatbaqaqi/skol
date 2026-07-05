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
from typing import Any, Dict, List


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
    # Ascomycete / basidiomycete anatomical Latin (original set)
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
    # Slime mold (Myxomycota) anatomical Latin — added 2026-07-05
    # per taxon_67cc93d2 evidence.  Fungi-agnostic goal violated
    # locally as a stopgap; see docs/plans/lotl-detector.md.
    'capillitium', 'capillitia', 'capillities',
    'pseudocapillitium',
    'columella', 'columellae', 'columellula',
    'sporocarpium', 'sporocarpia',
    'aethalium', 'aethalia',
    'plasmodiocarpium', 'plasmodiocarpa',
    'sporangium', 'sporangia',
    'plasmodium', 'plasmodia',
    'hypothallus', 'calyculus', 'calyculata',
    # General mycological Latin morphology terms (fungi-agnostic)
    # that appear across clades and slipped past the suffix
    # heuristic.  Enriches signal for any Latin diagnosis.
    'fugax', 'valde', 'gradatim', 'usque', 'quam',
    'niger', 'nigra', 'nigrum',
    'crassa', 'crassum', 'crassus', 'crassior',
    'ramosum', 'ramosa', 'ramosus', 'ramosior',
    'anastomosans', 'attinens', 'macrescens',
    'constitutum', 'constituta', 'constitutus',
    'brunneus', 'brunnea', 'brunneum', 'brunneis',
    'angustus', 'angusta', 'angustum', 'angustis',
    'pallido', 'pallidus', 'pallida', 'pallidum',
    'apicem', 'partem', 'tertiam',
})
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


# Diagnosis / Description header terminators.  Three forms accepted:
#
#   1. Punctuation `:`, `-`, `–` (en-dash), `—` (em-dash) —
#      the standard colon/dash form.  taxon_8f93bded uses em-dash.
#   2. Period followed by whitespace and a capital letter
#      (M2 refinement, taxon_d65547ed's `Description. Colonies on
#      PDA…` form).  Requires the capital-letter lookahead to
#      distinguish header form from prose ending in
#      `…description. the next sentence…`.
#   3. One-or-more U+FFFD replacement characters (M2 refinement,
#      taxon_e0d2e4bb / taxon_95dbdfb9 shape where OCR noise sits
#      between the label and the content).  U+FFFD is
#      pipeline-specific (Python decode(errors='replace') artefact);
#      if we swap the ingest decoder, this pattern needs revisiting.
_DIAG_HEADER_RE = re.compile(
    r'\bDiagnosis\s*(?:[-–—:]|\.\s+(?=[A-Z])|�+)'
)
_DESC_HEADER_RE = re.compile(
    r'\bDescription\s*(?:[-–—:]|\.\s+(?=[A-Z])|�+)'
)


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


# §6 idea #3 aggregate watchlist.  Section-header keywords that
# each fire a merge signal when repeated within one description.
# EXCLUDES `Description` and `Diagnosis` (dedicated counters
# handle those and firing here too would double-count the same
# merge signal).  EXCLUDES `Cultural characteristics`,
# `Culture characteristics`, and `Colonies on` (substrate-specific
# subtypes — a single species on multiple culture media
# legitimately repeats these; taxon_b9a6232 false-positive
# prevention).  Ordered from longest to shortest so multi-word
# keywords match before their shorter prefixes.
_SECTION_HEADER_WATCHLIST = (
    'Description and illustration',
    'Illustration',
    'Observations',
    'Etymology',
    'Habitat',
    'Holotype',
    'Type',
)


def _watchlist_header_regex(keyword: str) -> 're.Pattern[str]':
    """Build a header-terminator regex for a watchlist keyword.
    Same three-form terminator (colon/dash, period + capital,
    U+FFFD run) as `_DIAG_HEADER_RE` / `_DESC_HEADER_RE`."""
    return re.compile(
        rf'\b{re.escape(keyword)}\s*(?:[-–—:]|\.\s+(?=[A-Z])|�+)'
    )


_WATCHLIST_HEADER_REGEXES = {
    kw: _watchlist_header_regex(kw)
    for kw in _SECTION_HEADER_WATCHLIST
}


def count_repeated_section_headers(text: str) -> int:
    """Count DISTINCT watchlist section-header keywords that
    appear at least twice in ``text``.  §6 idea #3 aggregate
    detector.

    Independent of ``count_description_headers`` /
    ``count_diagnosis_headers`` — those keywords are excluded
    from the watchlist so both signals don't fire on the same
    merge.  Fires ``§6:multi_section_header`` in
    ``predicted_issues``.

    Example: taxon_592128a8 has three ``Observations:`` blocks
    → this returns 1 (one distinct header keyword repeated).
    taxon_2a9d07e6 has two ``Description and illustration:``
    citations AND two implicit species-boundary markers
    of another type → returns 2.
    """
    if not text:
        return 0
    distinct_repeated = 0
    for regex in _WATCHLIST_HEADER_REGEXES.values():
        if len(regex.findall(text)) >= 2:
            distinct_repeated += 1
    return distinct_repeated


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

    Reused for the diagnosis field via the
    ``diag_starts_mid_sentence`` key in ``treatment_signals``
    — the same head-clip predicate applies regardless of
    which field the text came from.
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


# Sentence-final characters that indicate a description reached
# its natural end.  Ellipsis (…) is stylistic but common at
# genuine tails; treat as terminal to avoid flagging legitimate
# uses.
_SENTENCE_FINAL_CHARS = frozenset('.?!…')

# Mid-word hyphen at description tail — the taxon_9ecad903
# canonical shape (`cinnamon or red-`).  Strong signal that a
# page or paragraph break wasn't handled during extraction.
# Requires a lowercase letter before the hyphen so that
# proper hyphenated compounds ending in a period
# (`reddish-brown.`) don't match.
_MID_WORD_HYPHEN_TAIL_RE = re.compile(r'[a-z]-\s*$')


def tail_clipped(text: str) -> bool:
    """True if the description's tail looks truncated (§10
    tail-clip pattern).

    Two OR'd sub-signals:

      * **Mid-word hyphen**: text ends with a lowercase letter
        followed by a hyphen (`[a-z]-\\s*$`).  The taxon_9ecad903
        canonical case — page-break marker the extractor
        preserved without joining.
      * **No sentence-final punctuation**: after stripping
        trailing whitespace, the last non-whitespace character
        is neither `.` nor `?` nor `!` nor `…`.  Catches
        taxon_ae45a05e's `Pileus 5-10 mm, … Pil` tail (just
        runs out) and taxon_23d479f4's two-ended clip.

    Closing brackets / parens followed by sentence-final
    punctuation are handled correctly because we scan back to
    the last non-whitespace char — `(Fig. 2).` ends with `.`.
    """
    if not text:
        return False
    stripped = text.rstrip()
    if not stripped:
        return False
    if _MID_WORD_HYPHEN_TAIL_RE.search(stripped):
        return True
    last = stripped[-1]
    return last not in _SENTENCE_FINAL_CHARS


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


def latin_between_english(
    text: str, threshold: float = 0.35,
) -> bool:
    """True if the description contains a Latin paragraph
    with English paragraphs on BOTH sides (E → L → E
    ordering).  §6 idea #1(b) — the operator's 2026-07-03
    correction: a Latin block sandwiched by English is a
    merge signal even when only ONE Latin block exists
    (below the ``latin_block_count >= 2`` threshold).

    Rationale: normal taxonomic-paper structure puts Latin
    FIRST (or lets the two languages live in separate
    labelled sections).  Latin appearing mid-description
    with English on both sides means the assembler
    collapsed adjacent species' content across a Latin
    diagnosis that should have anchored one of them.
    Canonical case: taxon_9ecad903.

    Composes with ``latin_block_count`` — both can fire
    on the same treatment.  Multi-Latin patterns
    (``L → E → L``) get ``§6:latin_alt``; the sandwich
    (``E → L → E``) gets ``§6:latin_ele``; the doubled
    sandwich (``E → L → E → L → E``) fires both.

    Limitation: requires paragraph structure.  A
    single-paragraph description with intra-paragraph
    mixed language can't be detected — no block boundaries
    to score.  Empirically the failure mode preserves
    paragraph breaks between merged species' content, so
    this covers the observed cases.
    """
    if not text:
        return False
    labels = []
    for p in re.split(r'\n\s*\n', text):
        if not p.strip():
            continue
        labels.append('L' if _latin_ratio(p) >= threshold else 'E')
    # Merge consecutive same-labels: e.g., [L, L, E, L] → [L, E, L].
    merged: List[str] = []
    for lab in labels:
        if not merged or merged[-1] != lab:
            merged.append(lab)
    if len(merged) < 3:
        return False
    # Interior 'L' (not first, not last) means English on both sides.
    return any(lab == 'L' for lab in merged[1:-1])


# ---------------------------------------------------------------------------
# Composed helpers over a full Treatment doc
# ---------------------------------------------------------------------------


def treatment_signals(
    treatment: Dict[str, Any],
    *,
    authored_binomial_in_desc: Any = None,
) -> Dict[str, Any]:
    """Compute all triage signals for a Treatment doc, return
    as a flat dict suitable for CSV columns.

    The returned dict is intentionally verbose — every derived
    value is a column so operators can eyeball the raw signals,
    not just the summary verdict.

    ``authored_binomial_in_desc`` is an OPTIONAL keyword arg for
    the gnfinder+gnparser-based §6:authored_binomial signal.
    Pass in the pre-computed boolean from
    ``gn_client.authored_binomial_in_text`` — this function stays
    pure Python (no HTTP) so unit tests don't need mocked
    services.  ``None`` (default) is treated as "not evaluated"
    → False in the output dict.
    """
    desc = treatment.get('description') or ''
    diag = treatment.get('diagnosis') or ''
    return {
        'desc_length': len(desc),
        'diag_length': len(diag),
        'n_diagnosis_headers': count_diagnosis_headers(desc),
        'n_description_headers': count_description_headers(desc),
        'n_repeated_section_headers':
            count_repeated_section_headers(desc),
        'n_sp_nov': count_sp_nov(desc),
        'n_key_couplets': count_key_couplets(desc),
        'desc_starts_mid_sentence':
            desc_starts_mid_sentence(desc),
        'latin_block_count': latin_block_count(desc),
        'latin_between_english':
            latin_between_english(desc),
        'mid_body_description_header':
            mid_body_description_header(desc),
        'tail_clipped': tail_clipped(desc),
        # Reuse the same head-clip predicate on the diagnosis
        # field.  Gated on non-empty implicitly by the
        # predicate itself (empty text returns False).
        'diag_starts_mid_sentence':
            desc_starts_mid_sentence(diag),
        # §6 idea #2 (gnfinder+gnparser): supplied by the caller
        # via keyword arg.  None → False in the output; a bool
        # passes through.
        'authored_binomial_in_desc':
            bool(authored_binomial_in_desc),
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
    if signals.get('tail_clipped'):
        flags.append('§10:tail_clip')
    if signals.get('diag_starts_mid_sentence'):
        flags.append('§10:diag_head_clip')
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
    if signals.get('n_repeated_section_headers', 0) >= 1:
        flags.append('§6:multi_section_header')
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
    if signals.get('latin_between_english'):
        flags.append('§6:latin_ele')
    if signals.get('authored_binomial_in_desc'):
        flags.append('§6:authored_binomial')
    if merge_metric >= merge_threshold:
        flags.append(f'§6:merge_metric={merge_metric}')
    return '|'.join(flags)


__all__ = (
    'count_diagnosis_headers',
    'count_description_headers',
    'count_repeated_section_headers',
    'count_sp_nov',
    'count_key_couplets',
    'desc_starts_mid_sentence',
    'latin_between_english',
    'latin_block_count',
    'mid_body_description_header',
    'tail_clipped',
    'treatment_signals',
    'predicted_issues',
)
