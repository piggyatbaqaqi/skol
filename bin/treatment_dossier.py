#!/usr/bin/env python3
"""Render the diagnostic context for a treatment, read-only.

The brat surface shows `=== description ===` and prose.  Every
pathology diagnosed during round-4 review came from data it does not
show — the layout label each span carried, the paragraph numbers, the
blocks that fell between spans, `merge_metric`, the triage flags, the
source document's identity, the sibling treatments.  Reviewers were
asked to infer all of that, which is why findings cost ~20 minutes
each.

This is the companion tab to brat, not a replacement for it.  **The
brat `.txt` is deliberately left alone**: `brat_export` renders the
synthetic doc through `render()` and `brat_ingest` re-renders it to
translate offsets back, so changing that format would shift every
offset and invalidate existing `.ann` files — round 5's included,
mid-flight.  Context text inside an annotation surface is also
annotatable text, and reviewers would end up labelling material that
is not part of the treatment.

Usage::

    bin/treatment_dossier --experiment production_v4 \\
        --doc-id taxon_fdbd1b53... --output-dir /tmp/dossiers

    # terminal, and the form T3a's merge-suspect table is built from
    bin/treatment_dossier --experiment production_v4 --format text \\
        --doc-id taxon_a,taxon_b

    # streaming, per T0d
    bin/treatment_dossier --experiment production_v4 --doc-id - \\
        --format text < ids.txt

Writes nothing to CouchDB.  Ever.

Skeleton only — implementation follows test confirmation (CLAUDE.md).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatments_to_structured.dossier import (  # noqa: E402
    Dossier,
)

#: Prose fields rendered in full, in the order a treatment reads.
PROSE_FIELDS = (
    'treatment', 'diagnosis', 'description', 'notes', 'etymology',
    'type_designation', 'materials_examined', 'biology', 'key',
    'figure_captions', 'distribution',
)


@dataclass
class DossierView:
    """A dossier plus everything around it that needs rendering."""

    dossier: Dossier
    treatment: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, Any] = field(default_factory=dict)
    status: Dict[str, Any] = field(default_factory=dict)
    siblings: List[Dict[str, Any]] = field(default_factory=list)
    merge_metric: Optional[int] = None
    flags: str = ''


def build_view(
    treatment: Dict[str, Any],
    ann_text: str,
    *,
    source: Optional[Dict[str, Any]] = None,
    status: Optional[Dict[str, Any]] = None,
    siblings: Optional[List[Dict[str, Any]]] = None,
) -> DossierView:
    """Assemble a view.  Pure: takes documents, returns a value.

    Every argument beyond the treatment is optional, because the
    dossier has to render for treatments that have no status doc, no
    resolvable source, or no siblings — p2b's 35 482 are exactly the
    ones worth looking at and exactly the ones missing the most.
    """
    raise NotImplementedError


def render_text(view: DossierView) -> str:
    """Plain-text rendering, for the terminal and for piping.

    Kept alongside the HTML because T3a's merge-suspect table is a
    *view over this renderer* rather than a separate throwaway script,
    and a markdown table is built from text, not from a web page.
    """
    raise NotImplementedError


def render_html(view: DossierView) -> str:
    """Self-contained HTML — no external CSS, JS or fonts.

    It is opened from a ``file://`` URL in the tab beside brat, where
    nothing else will load.
    """
    raise NotImplementedError


def main() -> int:
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
