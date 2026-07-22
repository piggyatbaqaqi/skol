"""Resolver for Treatment ``source_anchors`` → user-facing deep links.

Trello #401 Phase 1 Commit C.  Central authority for the priority
ordering and per-kind URL construction so no client (React,
downstream API user, cron report) has to reimplement the policy.

Contract:

  resolve_anchors(source_anchors, ingest) -> List[Dict]

Input:
  - ``source_anchors``: the raw list emitted by ``Treatment.as_row()``.
    Values are strings (matches the ``MapType(String, String)`` Spark
    schema).  Storage order is not part of the extractor's contract;
    this function imposes the priority policy.
  - ``ingest``: the Treatment's ``ingest`` dict.  Used to construct
    URLs for anchors whose payload doesn't carry a host-specific URL
    (only ``pdf`` today, which needs ``ingest['pdf_url']`` when
    available; the ``plazi`` / ``jats_section`` / ``mycobank`` /
    ``arpha`` resolvers use fixed per-host URL templates).

Output:
  - List of ``{"kind", "href", "label"}`` dicts, sorted by
    ``_KIND_PRIORITY``.  Only anchors whose kind is in
    ``_LINKABLE_KINDS`` are included — Tier-2 kinds (``arpha``,
    ``mycobank``) are dropped from the render output while still
    being persisted upstream.  When credentials or content quality
    change, add the kind to ``_LINKABLE_KINDS`` here; no re-extract
    required.

See ``docs/source-anchors.md`` for the design rationale and the
anchor-kind inventory.
"""

from typing import Any, Dict, List, Mapping, Optional


# Lower priority number → shown first.  A single source of truth for
# the anchor ordering policy.  Change this to reorder all consumers.
_KIND_PRIORITY: Dict[str, int] = {
    'pdf': 10,
    'plazi': 20,
    'jats_section': 30,
    'mycobank': 90,
    'arpha': 100,
}

# Kinds actually rendered as links right now.  Tier-2 kinds
# (``arpha``, ``mycobank``) are persisted but not rendered — see
# docs/source-anchors.md for why (ARPHA needs an app key, MycoBank
# pages are content-thin).  Promote a kind here when it becomes
# usable end-to-end.
_LINKABLE_KINDS = frozenset({'pdf', 'plazi', 'jats_section'})


def _pdf_href(anchor: Mapping[str, str], ingest: Mapping[str, Any]) -> Optional[str]:
    """Build a ``<pdf_url>#page=<page>`` fragment link.  Returns
    None when the ingest doc carries no ``pdf_url`` — the anchor
    payload alone can't construct an absolute URL."""
    pdf_url = ingest.get('pdf_url') if ingest else None
    if not pdf_url:
        return None
    page = anchor.get('page')
    if not page:
        return str(pdf_url)
    return f"{pdf_url}#page={page}"


def _plazi_href(anchor: Mapping[str, str], _ingest: Mapping[str, Any]) -> Optional[str]:
    """Build a Plazi TreatmentBank per-UUID URL.  Verified 2026-07-10
    to resolve free (no auth)."""
    uuid = anchor.get('uuid')
    if not uuid:
        return None
    return f"https://treatment.plazi.org/id/{uuid}"


def _jats_section_href(
    anchor: Mapping[str, str], ingest: Mapping[str, Any],
) -> Optional[str]:
    """Build the article HTML URL with a section-id fragment.

    Prefers ``ingest['url']`` (the human-facing article URL — e.g.
    ``https://mycokeys.pensoft.net/article/130565/``) so this works
    for any host, not just Pensoft.  Falls back to ``https://doi.org/
    <doi>`` if no ingest URL is present; browsers will follow the
    DOI redirect and preserve the fragment.  Returns None when
    neither is available.
    """
    section_id = anchor.get('section_id')
    if not section_id:
        return None
    if ingest:
        url = ingest.get('url')
        if url:
            base = url if url.endswith('/') else f"{url}/"
            return f"{base}#{section_id}"
    doi = anchor.get('doi')
    if doi:
        return f"https://doi.org/{doi}#{section_id}"
    return None


_HREF_BUILDERS = {
    'pdf': _pdf_href,
    'plazi': _plazi_href,
    'jats_section': _jats_section_href,
}


def _pdf_label(anchor: Mapping[str, str]) -> str:
    label = anchor.get('label')
    page = anchor.get('page')
    if label and label != page:
        return f"PDF page {label}"
    return f"PDF page {page}" if page else "PDF"


def _plazi_label(_anchor: Mapping[str, str]) -> str:
    return "Open at Plazi"


def _jats_section_label(_anchor: Mapping[str, str]) -> str:
    return "Open article at treatment"


_LABEL_BUILDERS = {
    'pdf': _pdf_label,
    'plazi': _plazi_label,
    'jats_section': _jats_section_label,
}


def resolve_anchors(
    source_anchors: List[Dict[str, Any]],
    ingest: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Turn a raw ``source_anchors`` list into rendered ``deep_links``.

    Sorts by ``_KIND_PRIORITY``, drops non-linkable kinds, computes
    href + label per remaining anchor.  Anchors whose payload can't
    produce a valid href (e.g. ``pdf`` without an ingest ``pdf_url``)
    are silently dropped — the alternative would be surfacing broken
    links to end users, which is worse.

    Returns a list of ``{"kind", "href", "label"}`` dicts, in
    priority order.  Callers can render each as ``<a href={href}>
    {label}</a>`` with no per-kind knowledge.
    """
    if not source_anchors:
        return []
    ingest_map = ingest or {}
    sorted_anchors = sorted(
        source_anchors,
        key=lambda a: _KIND_PRIORITY.get(a.get('kind', ''), 999),
    )
    deep_links: List[Dict[str, str]] = []
    for anchor in sorted_anchors:
        kind = anchor.get('kind', '')
        if kind not in _LINKABLE_KINDS:
            continue
        href_builder = _HREF_BUILDERS.get(kind)
        label_builder = _LABEL_BUILDERS.get(kind)
        if href_builder is None or label_builder is None:
            continue
        href = href_builder(anchor, ingest_map)
        if not href:
            continue
        deep_links.append({
            'kind': kind,
            'href': href,
            'label': label_builder(anchor),
        })
    return deep_links
