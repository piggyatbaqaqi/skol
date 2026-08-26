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
"""

import argparse
import html
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatments_to_structured.dossier import (  # noqa: E402
    Dossier,
    build_dossier,
)
from treatments_to_structured.merge_metric import (  # noqa: E402
    treatment_merge_metric,
)
from env_config import common_parser, get_env_config  # noqa: E402
from llm_annotate_features import read_treatment_ids  # noqa: E402
from span_resolver import (  # noqa: E402
    _attachment_text,
    coordinate_space,
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
    #: The article.txt.ann text the spans index into, so the renderer
    #: can show each span's own source on hover.
    ann_text: str = ''


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
    try:
        metric: Optional[int] = treatment_merge_metric(treatment)
    except Exception:              # noqa: BLE001 - a view must render
        metric = None
    return DossierView(
        dossier=build_dossier(treatment, ann_text),
        treatment=treatment,
        source=source or {},
        status=status or {},
        siblings=list(siblings or []),
        merge_metric=metric,
        ann_text=ann_text,
    )


def render_text(view: DossierView) -> str:
    """Plain-text rendering, for the terminal and for piping.

    Kept alongside the HTML because T3a's merge-suspect table is a
    *view over this renderer* rather than a separate throwaway script,
    and a markdown table is built from text, not from a web page.
    """
    d = view.dossier
    out: List[str] = [f'=== {d.treatment_id} ===']
    src = view.source
    if src:
        bits = [str(src.get(k)) for k in ('journal', 'volume', 'year')
                if src.get(k)]
        out.append('source: ' + ' '.join(bits)
                   + (f"  doi:{src['doi']}" if src.get('doi') else ''))
    meta = [f'merge_metric={view.merge_metric}']
    if view.status.get('status'):
        meta.append(f"bootstrap={view.status['status']}")
    if view.status.get('round') is not None:
        meta.append(f"round={view.status['round']}")
    if view.flags:
        meta.append(f'flags={view.flags}')
    out.append('  '.join(meta))

    out.append('')
    out.append('--- spans (paragraph, offsets, layout label) ---')
    for sp in d.spans:
        labels = ','.join(_labels_of(d, sp)) or '?'
        out.append(f'  {sp.field:<20} para {sp.paragraph}  '
                   f'[{sp.start}:{sp.end}]  <{labels}>')
    if not d.spans:
        out.append('  (none)')

    out.append('')
    out.append('--- gaps: blocks between consecutive spans ---')
    for g in d.gaps:
        extra = []
        if g.n_furniture:
            extra.append(f'{g.n_furniture} furniture hidden')
        if g.n_omitted:
            extra.append(f'{g.n_omitted} more not shown')
        tail = f"   [{', '.join(extra)}]" if extra else ''
        out.append(f'  {g.after.field}@{g.after.paragraph} -> '
                   f'{g.before.field}@{g.before.paragraph}{tail}')
        for b in g.blocks:
            out.append(f'      [{b.label}] {b.head[:96]}')
    if not d.gaps:
        out.append('  (none)')

    out.append('')
    out.append('--- fields ---')
    for f in PROSE_FIELDS:
        val = view.treatment.get(f)
        if val and str(val).strip():
            out.append(f'  {f} ({len(str(val))}):')
            out.append('    ' + str(val).strip().replace('\n', '\n    '))

    if view.siblings:
        out.append('')
        out.append('--- siblings from the same source document ---')
        for sib in view.siblings:
            name = str(sib.get('treatment') or '').strip().replace('\n', ' ')
            out.append(f"  {str(sib.get('_id'))[:22]}  {name[:64]}")
    return '\n'.join(out) + '\n'


def _span_text(d: Dossier, span: Any) -> str:
    """The source text a span covers, taken from the blocks it overlaps.

    NOT a raw slice of the ``.ann`` at the span's own offsets: those
    bound the whole block, so slicing yields ``[@…#Label*]`` wrappers
    and, for a multi-block span, silently runs them together.

    A span crossing a block boundary happens in 0.7 % of treatments and
    is itself a finding -- the extractor joined material the layout
    pass had separated -- so those get a per-block label to make the
    join visible.  Single-block spans, the other 99.3 %, stay clean.
    """
    covered = [b for b in d.blocks
               if span.start < b.end and span.end > b.start]
    if len(covered) <= 1:
        return covered[0].text if covered else ''
    return '\n\n'.join(f'#{b.label}*]\n{b.text}' for b in covered)


def _labels_of(d: Dossier, span: Any) -> List[str]:
    """Layout labels covering one span, from the assembled blocks."""
    return [b.label for b in d.blocks
            if span.start < b.end and span.end > b.start]


def render_html(view: DossierView) -> str:
    """Self-contained HTML — no external CSS, JS or fonts.

    It is opened from a ``file://`` URL in the tab beside brat, where
    nothing else will load.
    """
    e = html.escape
    d = view.dossier
    gap_by_after = {(g.after.start, g.after.end): g for g in d.gaps}

    def _gap_html(gap: Any) -> str:
        notes = []
        if gap.blocks:
            notes.append(f'{len(gap.blocks)} block(s): '
                         + ', '.join(sorted({b.label for b in gap.blocks})))
        if gap.n_furniture:
            notes.append(f'{gap.n_furniture} furniture hidden')
        if gap.n_omitted:
            notes.append(f'{gap.n_omitted} further not shown')
        items = ''.join(
            f'<li><span class="lab">{e(b.label)}</span>'
            f'<pre>{e(b.text[:4000])}</pre></li>' for b in gap.blocks)
        return (
            f'<details class="gap"><summary>&#9656; gap before '
            f'{e(gap.before.field)}@{e(str(gap.before.paragraph))} '
            f'&mdash; {e("; ".join(notes) or "empty")}</summary>'
            f'<ul>{items}</ul></details>')

    flow: List[str] = []
    shown: set = set()
    for sp in d.spans:
        labels = ', '.join(_labels_of(d, sp)) or '?'
        body = e(_span_text(d, sp))
        flow.append(
            f'<div class="span"><span class="fld">{e(sp.field)}</span>'
            f'<span class="muted"> paragraph {e(str(sp.paragraph))} '
            f'&nbsp; {sp.start}:{sp.end} &nbsp;</span>'
            f'<span class="lab">{e(labels)}</span>'
            f'<div class="txt">{body}</div></div>')
        gap = gap_by_after.get((sp.start, sp.end))
        if gap is None:
            continue
        shown.add(id(gap))
        flow.append(_gap_html(gap))
    # A gap whose `after` matched no span would otherwise disappear.
    # Show it at the end rather than lose it.
    for gap in d.gaps:
        if id(gap) not in shown:
            flow.append(_gap_html(gap))

    fields_html: List[str] = []
    for f in PROSE_FIELDS:
        val = view.treatment.get(f)
        if val and str(val).strip():
            fields_html.append(
                f'<h3>{e(f)} <span class="muted">({len(str(val))} '
                f'chars)</span></h3><pre>{e(str(val).strip())}</pre>')
    sibs = ''.join(
        f'<li>{e(str(s.get("_id"))[:22])} '
        f'{e(str(s.get("treatment") or "").strip()[:80])}</li>'
        for s in view.siblings)
    src = view.source
    src_line = e(' '.join(str(src.get(k)) for k in
                          ('journal', 'volume', 'year') if src.get(k)))
    meta = [f'merge_metric = {view.merge_metric}']
    if view.status.get('status'):
        meta.append(f'bootstrap = {view.status["status"]}')
    if view.status.get('round') is not None:
        meta.append(f'round = {view.status["round"]}')
    if view.flags:
        meta.append(f'flags = {view.flags}')
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{e(d.treatment_id)}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:60rem}}
h1{{font-size:1.1rem;font-family:monospace}}
table{{border-collapse:collapse;width:100%}}
td{{border-bottom:1px solid #ddd;padding:.2rem .5rem;vertical-align:top}}
.lab{{font-family:monospace;background:#eef;padding:0 .3rem}}
.fld{{font-weight:600}}
.span{{position:relative;padding:.25rem .5rem;border-bottom:1px solid #eee}}
.span:hover{{background:#f4f8ff}}
.span .txt{{display:none}}
.span:hover .txt{{display:block;position:absolute;left:1rem;top:1.7rem;
z-index:9;max-width:52rem;max-height:20rem;overflow:auto;
white-space:pre-wrap;background:#fffbe6;border:1px solid #ccb;
padding:.5rem;box-shadow:0 2px 8px rgba(0,0,0,.2)}}
.gap{{border-left:3px solid #c33;padding:.15rem .8rem;margin:.2rem 0 .2rem 1rem;
background:#fff8f8}}
.gap summary{{cursor:pointer;color:#a22}}
.gap ul{{margin:.3rem 0;padding-left:1.2rem}}
.gap pre{{margin:.2rem 0}}
.muted{{color:#777;font-weight:normal}}
pre{{white-space:pre-wrap;background:#f7f7f7;padding:.6rem;margin:.2rem 0}}
</style></head><body>
<h1>{e(d.treatment_id)}</h1>
<p class="muted">{src_line}</p>
<p>{e('   '.join(meta))}</p>
<h2>Spans <span class="muted">&mdash; hover for the source text;
gaps open on click</span></h2>
{''.join(flow) or '<p class="muted">(none)</p>'}
<h2>Fields</h2>{''.join(fields_html)}
{'<h2>Siblings</h2><ul>' + sibs + '</ul>' if sibs else ''}
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--format', choices=('html', 'text'), default='html',
        help='html writes one file per treatment into --output-dir; '
             'text writes to stdout.  Default: html.',
    )
    parser.add_argument(
        '--output-dir', default=None, metavar='DIR',
        help='Where html pages are written.  Required for --format html.',
    )
    parser.add_argument(
        '--no-siblings', action='store_true',
        help='Skip the sibling lookup, which scans the treatments DB '
             'once per source document.',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)
    experiment = config.get('experiment_name')
    if not experiment:
        print('error: --experiment is required', file=sys.stderr)
        return 2
    try:
        ids = read_treatment_ids(
            config.get('doc_ids'), sys.stdin,
            stdin_isatty=sys.stdin.isatty(),
        )
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    out_dir: Optional[Path] = None
    if args.format == 'html':
        if not args.output_dir:
            print('error: --output-dir is required with --format html',
                  file=sys.stderr)
            return 2
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )
    exp = server['skol_experiments'][experiment]
    dbs = exp.get('databases') or {}
    treatments = server[dbs['treatments_prose']]
    ingest = server[dbs.get('ingest') or config['ingest_db_name']]
    status_name = dbs.get('features_status')
    status_db = (server[status_name]
                 if status_name and status_name in server else None)

    # Fetch every requested treatment first, so the sibling index costs
    # ONE scan for the whole run rather than one per treatment.  At
    # ~11 s a scan that is the difference between 11 s and 5.5 min for
    # T3a's 30-treatment table.
    docs: List[Dict[str, Any]] = []
    for tid in ids:
        try:
            docs.append(dict(treatments[tid]))
        except Exception as exc:                     # noqa: BLE001
            print(f'  skipping {tid}: {exc}', file=sys.stderr)
    sib_index: Dict[str, List[Dict[str, Any]]] = {}
    if not args.no_siblings:
        wanted = {(d.get('ingest') or {}).get('_id') for d in docs}
        wanted.discard(None)
        if wanted:
            sib_index = _sibling_index(treatments, wanted)

    written = 0
    for doc in docs:
        tid = doc['_id']
        try:
            ann_text = _attachment_text(coordinate_space(doc), server)
        except Exception as exc:                     # noqa: BLE001
            # A dossier without the .ann still shows the fields and the
            # metrics, which beats refusing to render.
            print(f'  {tid}: no annotation text ({exc})', file=sys.stderr)
            ann_text = ''
        src_id = (doc.get('ingest') or {}).get('_id')
        source: Dict[str, Any] = {}
        if src_id:
            try:
                source = dict(ingest[src_id])
            except Exception:                        # noqa: BLE001
                pass
        status: Dict[str, Any] = {}
        if status_db is not None:
            try:
                status = dict(status_db[tid])
            except Exception:                        # noqa: BLE001
                pass
        siblings = [s for s in sib_index.get(src_id or '', [])
                    if s['_id'] != tid]
        view = build_view(doc, ann_text, source=source, status=status,
                          siblings=siblings)
        if out_dir is not None:
            path = out_dir / f'{tid}.html'
            path.write_text(render_html(view), encoding='utf-8')
            print(f'  {path}', file=sys.stderr)
        else:
            print(render_text(view))
        written += 1
    print(f'{written} dossier(s)', file=sys.stderr)
    return 0


def _sibling_index(
    treatments: Any, src_ids: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Map each wanted source document to its treatments, in order.

    One pass over the treatments DB for however many documents are
    wanted.  The question "what else came out of this article, and in
    what order" is what placed the Rhodoveronaea header in
    taxon_fd50457a and identified the four treatments lost from rounds
    2 and 3 -- but it is a full scan, so it happens once.
    """
    wanted = set(src_ids)
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in wanted}
    for row in treatments.view('_all_docs', include_docs=True):
        if row.id.startswith('_'):
            continue
        doc = row.doc or {}
        src = (doc.get('ingest') or {}).get('_id')
        if src not in wanted:
            continue
        paras = [
            int(x['paragraph_number'])
            for f in doc if f.endswith('_spans') and doc[f]
            for x in doc[f]
            if str(x.get('paragraph_number', '')).lstrip('-').isdigit()
        ]
        out[src].append({'_id': row.id,
                         'treatment': doc.get('treatment'),
                         'paragraph': min(paras) if paras else None})
    for rows in out.values():
        rows.sort(key=lambda d: (d['paragraph'] is None,
                                 d['paragraph'] or 0))
    return out


if __name__ == '__main__':
    sys.exit(main())
