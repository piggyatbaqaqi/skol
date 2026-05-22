"""Pipeline state shared across components within a single doc run.

A :class:`PipelineState` is the mutable per-document carrier the
dispatcher hands to each :class:`ComponentInstance`'s ``run()`` call.
Components read attachments + properties from it and append their
contributions (section labels, spans).

After all components run, the assembler reads the merged contributions
and produces ``Treatment`` objects.

For the v3_buildout (extraction-pipeline Commit 1) the merge rules are
intentionally simple:

* **Labels** — highest-priority contributor wins entirely.  Future
  commits introduce range-aware merging when multiple labelers
  contribute to disjoint passages of the same doc.
* **Spans** — concatenated; downstream
  :func:`ingestors.spans.resolve_conflicts` handles overlap if
  needed.

Refer to docs/extraction_pipeline.md §"Output merging" for the
longer-term merge contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ingestors.spans import Span
from ingestors.yedda_tags import TaggedBlock


# ---------------------------------------------------------------------------
# Contribution records
# ---------------------------------------------------------------------------


@dataclass
class LabelContribution:
    """A labeler's offer of per-passage tag labels for a doc.

    Attributes:
        source: Component name that produced these labels.
        blocks: TaggedBlock list, in document order.
        priority: Higher wins on merge.  Convention: ``10`` for
            deterministic XML extractors, ``4`` for model-based
            labelers.
    """

    source: str
    blocks: List[TaggedBlock]
    priority: int = 0


@dataclass
class SpanContribution:
    """An entity detector's offer of spans for a doc."""

    source: str
    spans: List[Span]


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Per-doc mutable state passed through the dispatcher's component
    chain.

    Set up by the dispatcher with ``doc``, optionally ``couchdb_db``
    and ``redis_client`` for components that need live service
    handles, plus a ``config`` dict (env_config snapshot).

    Components mutate the state via :meth:`add_section_labels` and
    :meth:`add_spans`; the assembler reads
    :meth:`merged_section_labels` and :meth:`merged_spans` to assemble
    final ``Treatment`` records.
    """

    doc: Dict[str, Any] = field(default_factory=dict)
    props: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # Service handles populated by the dispatcher.  Optional so that
    # unit tests can construct a PipelineState without live infra.
    couchdb_db: Optional[Any] = None
    redis_client: Optional[Any] = None

    # Component outputs accumulate here.  Private to discourage
    # external direct mutation; use the add_* helpers.
    _label_contributions: List[LabelContribution] = field(default_factory=list)
    _span_contributions: List[SpanContribution] = field(default_factory=list)
    _attachment_cache: Dict[str, bytes] = field(default_factory=dict)

    # ---- Attachments --------------------------------------------------------

    def get_attachment(self, name: str) -> bytes:
        """Return the bytes of the named attachment.

        Looks in three places, in order:
          1. ``_attachment_cache`` populated by a prior ``get_attachment``.
          2. ``doc['_attachments'][name]`` if a test fixture pre-seeded
             a literal bytes/str value there.
          3. A live ``self.couchdb_db.get_attachment(doc_id, name)``
             call.

        Raises ``FileNotFoundError`` if the attachment is unavailable.
        """
        if name in self._attachment_cache:
            return self._attachment_cache[name]

        atts = self.doc.get("_attachments") or {}
        if name in atts:
            entry = atts[name]
            if isinstance(entry, (bytes, str)):
                data = entry.encode("utf-8") if isinstance(entry, str) else entry
                self._attachment_cache[name] = data
                return data
            # Fall through if entry is the CouchDB metadata dict — the
            # real bytes need a fetch from couchdb_db below.

        if self.couchdb_db is not None:
            doc_id = self.doc.get("_id")
            if doc_id:
                stream = self.couchdb_db.get_attachment(doc_id, name)
                if stream is not None:
                    data = stream.read()
                    self._attachment_cache[name] = data
                    return data

        raise FileNotFoundError(
            f"No attachment {name!r} on doc {self.doc.get('_id')!r}"
        )

    # ---- Contributions ------------------------------------------------------

    def add_section_labels(
        self,
        source: str,
        blocks: List[TaggedBlock],
        priority: int = 0,
    ) -> None:
        """Record a labeler's contribution.

        Args:
            source: Component name (for traceability + merge ordering).
            blocks: Per-passage TaggedBlock list in document order.
            priority: Higher wins on merge.
        """
        self._label_contributions.append(
            LabelContribution(source=source, blocks=blocks, priority=priority)
        )

    def add_spans(
        self,
        source: str,
        spans: List[Span],
    ) -> None:
        """Record an entity detector's contribution."""
        self._span_contributions.append(
            SpanContribution(source=source, spans=spans)
        )

    # ---- Merging ------------------------------------------------------------

    def merged_section_labels(self) -> List[TaggedBlock]:
        """Return the merged TaggedBlock list, applying the
        deterministic-first rule.

        For Commit 1 of the extraction pipeline only one labeler runs
        per doc (taxpub vs classifier), so the highest-priority
        contribution wins entirely.  Future commits will introduce
        range-aware merging.
        """
        if not self._label_contributions:
            return []
        winner = max(self._label_contributions, key=lambda c: c.priority)
        return list(winner.blocks)

    def merged_spans(self) -> List[Span]:
        """Return all span contributions concatenated.

        For Commit 1 the order matches the contribution order; future
        commits may add overlap resolution (see
        :func:`ingestors.spans.resolve_conflicts`).
        """
        spans: List[Span] = []
        for contrib in self._span_contributions:
            spans.extend(contrib.spans)
        return spans

    # ---- Provenance helpers (for tests + debugging) -------------------------

    def label_sources(self) -> List[str]:
        """Return the list of component names that contributed labels."""
        return [c.source for c in self._label_contributions]

    def span_sources(self) -> List[str]:
        """Return the list of component names that contributed spans."""
        return [c.source for c in self._span_contributions]
