#!/usr/bin/env python3
"""Extract a ``{pdf_relpath: metadata}`` mapping from the mykoweb
systematics tree.

Walks every ``*.html`` file under ``<site-root>/systematics/`` and
``<site-root>/systematics.html`` itself, pulls one row per ``<a
href>`` ending in ``.pdf``, classifies the row per source file, and
emits a flat JSON file.  Off-site (http://...) PDF links are
skipped — they have no local file to map.

The output is consumed by the mykoweb ingestor to tag each treatment
with its source journal / book / key.

Usage:
    bin/extract_mykoweb_pdf_metadata.py \\
        --site-root /data/skol/www/mykoweb.com \\
        --out       /data/skol/www/mykoweb.com/systematics_pdf_metadata.json \\
        --unparsed-out /tmp/mykoweb_unparsed.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

# ---------------------------------------------------------------------------
# Stage 1 — raw-row extraction
# ---------------------------------------------------------------------------


_YEAR_RE = re.compile(r'\((\d{4})\)')
_WS_RE = re.compile(r'\s+')


def _norm_ws(s: str) -> str:
    """Collapse runs of whitespace (including newlines) into single
    spaces.  Citation tails span multiple HTML lines in mykoweb;
    normalising at extraction time keeps the JSON output clean and
    keeps downstream regex anchors simple."""
    return _WS_RE.sub(' ', s)


def _li_text_around(li: Tag, anchor: Tag) -> tuple[str, str]:
    """Split text inside ``li``'s subtree into before / after the
    ``anchor`` element, in document order.  Skips descendants of
    ``anchor`` itself."""
    before_parts: List[str] = []
    after_parts: List[str] = []
    seen_anchor = False
    for desc in li.descendants:
        if desc is anchor:
            seen_anchor = True
            continue
        # Skip text that lives inside the anchor (that's the anchor's
        # own visible text, not the surrounding citation prose).
        if isinstance(desc, Tag):
            continue
        if not isinstance(desc, NavigableString):
            continue
        # NavigableString outside the anchor.
        if anchor in desc.parents:
            continue
        if seen_anchor:
            after_parts.append(str(desc))
        else:
            before_parts.append(str(desc))
    return _norm_ws(''.join(before_parts)), _norm_ws(''.join(after_parts))


def _first_bold_text(li: Tag) -> str:
    """First ``<strong>`` or ``<b>`` text anywhere inside the
    ``<li>`` (covers nested ``<em><strong>`` and ``<a><b>`` shapes).
    Empty string when none."""
    bold = li.find(['strong', 'b'])
    if bold is None:
        return ''
    return bold.get_text(' ', strip=True)


def _find_enclosing_li(anchor: Tag) -> Optional[Tag]:
    """Climb the tree to find the nearest enclosing ``<li>``, or
    None if the anchor isn't inside any list item."""
    return anchor.find_parent('li')


def extract_rows_from_html(
    html: str, source_html: str,
) -> List[Dict[str, Any]]:
    """Parse ``html`` and emit one row dict per local ``.pdf`` link.

    Off-site URLs (``http://`` / ``https://``) are skipped — we can
    only map PDFs that live on disk under the same site root.

    Each row carries enough raw context for downstream classifiers
    to decide kind / title / container without re-parsing the HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    rows: List[Dict[str, Any]] = []
    for anchor in soup.find_all('a', href=True):
        # BS4's type stubs say attrs may be a list; in practice for
        # standard HTML href is always a single string.  Coerce to
        # silence mypy without changing runtime behaviour.
        href = str(anchor.get('href', ''))
        # Strip query strings / fragments before extension check.
        bare_href = href.split('?', 1)[0].split('#', 1)[0]
        if not bare_href.lower().endswith('.pdf'):
            continue
        if href.lower().startswith(('http://', 'https://')):
            continue

        li = _find_enclosing_li(anchor)
        if li is not None:
            li_before, li_after = _li_text_around(li, anchor)
            strong_text = _first_bold_text(li)
            year_match: Optional[int] = None
            year_m = _YEAR_RE.search(li.get_text())
            if year_m:
                year_match = int(year_m.group(1))
        else:
            li_before, li_after, strong_text, year_match = '', '', '', None

        rows.append({
            'source_html':    source_html,
            'href':           href,
            'anchor_text':    anchor.get_text(' ', strip=True),
            'li_text_before': li_before,
            'li_text_after':  li_after,
            'strong_text':    strong_text,
            'year_match':     year_match,
        })
    return rows


# ---------------------------------------------------------------------------
# Stage 2a — citation-tail regex (the hard part of literature.html)
# ---------------------------------------------------------------------------


# CONTAINER VOL[(ISSUE)]: PAGES — pages accept hyphen or en-dash.
_CITATION_RE = re.compile(
    r'\b(?P<vol>\d+)(?:\((?P<issue>\d+)\))?:\s*(?P<pages>\d+[\-–]\d+)'
)

# A "sentence-break" period: ``. <Capital><lowercase>+`` — typical
# pattern at a real sentence boundary, distinct from abbreviation
# periods like ``Ann. Missouri Bot. Gard.`` where the chunk after
# the period is short.  Used by _refine_container_title to clip off
# interstitial prose from a too-long captured container.
_SENTENCE_BREAK_RE = re.compile(r'\.\s+(?=[A-Z][a-z])')

_CONTAINER_LONG_THRESHOLD = 50  # chars
_CONTAINER_MIN_TAIL = 10        # chars — refuse to clip to a stub


def _refine_container_title(container: Optional[str]) -> Optional[str]:
    """When the citation-regex capture runs on too long because the
    post-link tail had interstitial prose (e.g. the Peck row's
    ``State Botanist, with bibliographic locations cited and some of
    the most obvious synonyms given. Report of the State Botanist``),
    clip on the LAST sentence-style break — period followed by space
    followed by a Capitalized + lowercase word — and keep the trailing
    chunk if it's substantive.

    Short containers (the normal case) and containers whose
    trailing chunk after the last sentence break is too short
    (abbreviation periods like ``Ann. Missouri Bot. Gard.``) pass
    through unchanged.
    """
    if container is None or len(container) <= _CONTAINER_LONG_THRESHOLD:
        return container
    matches = list(_SENTENCE_BREAK_RE.finditer(container))
    if not matches:
        return container
    last = matches[-1]
    tail = container[last.end():].strip()
    if len(tail) < _CONTAINER_MIN_TAIL:
        return container
    return tail


def parse_citation_tail(text: str) -> Optional[Dict[str, Any]]:
    """Match ``CONTAINER VOL[(ISSUE)]: PAGES`` in the post-link text
    of a literature.html ``<li>``.  Returns the structured fields
    on a match, None on a miss (book / unstructured)."""
    if not text or not text.strip():
        return None
    m = _CITATION_RE.search(text)
    if not m:
        return None
    container = text[:m.start()].lstrip(' .,;:\n\t').rstrip()
    if not container:
        return None
    return {
        'container_title': _refine_container_title(container),
        'volume':          m.group('vol'),
        'issue':           m.group('issue'),
        'pages':           m.group('pages'),
    }


# ---------------------------------------------------------------------------
# Stage 2c — merging two records that resolve to the same PDF
# ---------------------------------------------------------------------------


# Kind specificity: lower index = more specific.  Used to decide
# which kind to keep when merging two records.
_KIND_ORDER = {
    'journal_article': 0,
    'book':            1,
    'journal':         2,
    'key':             3,
    'misc':            4,
}


def _is_meaningful(value: Any) -> bool:
    """Strings of pure whitespace / punctuation are not informative
    titles; treat them as missing when picking between two
    candidates."""
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip(' .,;:\n\t-_')
        return bool(stripped)
    return True


def merge_pdf_records(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine two records that resolved to the same PDF path.

    Picks the more-specific ``kind`` (journal_article > book > key /
    misc), then for every other field prefers the value that is
    meaningful (non-empty, not just punctuation) and — among
    meaningful strings — the longer of the two.  Either record may
    be missing fields; the result carries any field present in
    either.
    """
    a_kind = a.get('kind')
    b_kind = b.get('kind')
    if a_kind == b_kind:
        winning_kind = a_kind
    elif a_kind is None:
        winning_kind = b_kind
    elif b_kind is None:
        winning_kind = a_kind
    else:
        a_rank = _KIND_ORDER.get(a_kind, 99)
        b_rank = _KIND_ORDER.get(b_kind, 99)
        winning_kind = a_kind if a_rank <= b_rank else b_kind

    merged: Dict[str, Any] = {}
    if winning_kind is not None:
        merged['kind'] = winning_kind
    for key in set(a.keys()) | set(b.keys()):
        if key == 'kind':
            continue
        a_val = a.get(key)
        b_val = b.get(key)
        a_ok = _is_meaningful(a_val)
        b_ok = _is_meaningful(b_val)
        if a_ok and not b_ok:
            merged[key] = a_val
        elif b_ok and not a_ok:
            merged[key] = b_val
        elif a_ok and b_ok:
            # Both meaningful: prefer longer string, else a.
            if isinstance(a_val, str) and isinstance(b_val, str):
                merged[key] = a_val if len(a_val) >= len(b_val) else b_val
            else:
                merged[key] = a_val
        else:
            # Neither meaningful — preserve whatever's there.
            merged[key] = a_val if a_val is not None else b_val
    return merged


# ---------------------------------------------------------------------------
# Stage 2b — per-source classifiers
# ---------------------------------------------------------------------------


def classify_journals_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """journals.html: container = parent directory of PDF.

    100% reliable — the directory layout already encodes the answer.
    """
    parts = row['href'].split('/')
    container = parts[-2] if len(parts) >= 2 else ''
    volume_info = row['anchor_text']
    return {
        'kind':            'journal',
        'title':           f'{container} {volume_info}'.strip(),
        'container_title': container,
        'volume_info':     volume_info,
        'source_html':     row['source_html'],
    }


def classify_literature_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """literature.html: regex the post-link tail.  Match → journal
    article; miss → book."""
    citation = parse_citation_tail(row.get('li_text_after', ''))
    base = {
        'title':       row['anchor_text'],
        'author':      row.get('strong_text', ''),
        'year':        row.get('year_match'),
        'source_html': row['source_html'],
    }
    if citation:
        return {
            'kind':            'journal_article',
            'container_title': citation['container_title'],
            'volume':          citation['volume'],
            'issue':           citation['issue'],
            'pages':           citation['pages'],
            **base,
        }
    return {
        'kind':            'book',
        'container_title': None,
        **base,
    }


def classify_keys_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """keys.html: taxonomic key, not a citation — title only."""
    return {
        'kind':        'key',
        'title':       row['anchor_text'],
        'source_html': row['source_html'],
    }


def classify_misc_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """miscellanea.html / references.html: keep raw text for
    downstream hand-fixing."""
    return {
        'kind':          'misc',
        'title':         row['anchor_text'],
        'li_text_after': row.get('li_text_after', ''),
        'source_html':   row['source_html'],
    }


# ---------------------------------------------------------------------------
# Stage 2d — disk-only PDFs (CAF, misc/Omphalina, OldBooks, etc.)
# ---------------------------------------------------------------------------


# Roman → Arabic for the Omphalina-newsletter volume parser.
_ROMAN_VALUES: Dict[str, int] = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50,
    'C': 100, 'D': 500, 'M': 1000,
}


def _roman_to_int(s: str) -> Optional[int]:
    """Lenient Roman-to-Arabic conversion.  Returns None on empty
    input or any non-Roman character (rather than raising)."""
    if not s:
        return None
    s = s.upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        if ch not in _ROMAN_VALUES:
            return None
        v = _ROMAN_VALUES[ch]
        if v < prev:
            total -= v
        else:
            total += v
        prev = v
    return total if total > 0 else None


def pretty_title_from_stem(stem: str) -> str:
    """Filename stem → human-readable title (underscores → spaces,
    runs of whitespace collapsed, ends trimmed)."""
    return _norm_ws(stem.replace('_', ' ')).strip()


_OMPHALINA_RE = re.compile(r'^O-([IVXLCDM]+)-(\d+)$', re.IGNORECASE)


def parse_omphalina_filename(stem: str) -> Optional[Dict[str, Any]]:
    """``O-VIII-2`` → ``{volume: '8', issue: '2', title: 'Omphalina
    Vol. 8 No. 2'}``.  Returns None when the stem doesn't match the
    Omphalina naming convention."""
    m = _OMPHALINA_RE.match(stem)
    if not m:
        return None
    vol = _roman_to_int(m.group(1))
    if vol is None:
        return None
    issue = m.group(2)
    return {
        'volume': str(vol),
        'issue':  issue,
        'title':  f'Omphalina Vol. {vol} No. {issue}',
    }


# Disk-extraction rules.  Order matters — the first rule whose
# ``root`` is a prefix of the file's relpath wins, so the more
# specific paths must come before the less specific ones (e.g.
# ``CAF/PDF/FungaNordica`` before ``CAF/PDF``).  ``recursive=False``
# limits a rule to files directly in that directory (used to keep
# ``CAF/PDF`` from swallowing ``CAF/PDF/FungaNordica`` content).
_DISK_RULES: list = [
    {
        'root':             'CAF/protologue',
        'kind':             'misc',
        'container_title':  'California Fungi',
        'title_from':       'stem',
        'recursive':        True,
    },
    {
        'root':             'CAF/PDF/FungaNordica',
        'kind':             'book',
        'container_title':  'Funga Nordica',
        'title_from':       'after_dash',
        'recursive':        True,
    },
    {
        'root':             'CAF/PDF',
        'kind':             'book',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        False,  # exclude the FungaNordica subdir
    },
    {
        'root':             'CAF/keys',
        'kind':             'key',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        True,
    },
    {
        'root':             'misc/Omphalina',
        'kind':             'journal_article',  # overridden if parse fails
        'container_title':  'Omphalina',
        'title_from':       'omphalina',
        'recursive':        True,
    },
    {
        # Disk-only PDFs under ``systematics/literature/`` that
        # aren't linked from literature.html.  Order matters: this
        # rule runs *after* the HTML pass adds its records to
        # ``metadata``, and ``run()`` only writes a disk record
        # when the relpath isn't already present — so this rule
        # only contributes the stragglers, never overwrites the
        # HTML-derived records.
        'root':             'systematics/literature',
        'kind':             'book',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        True,
    },
    {
        # A separate top-level literature/ directory (distinct from
        # systematics/literature/).  A couple of monograph PDFs
        # live here.
        'root':             'literature',
        'kind':             'book',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        False,
    },
    {
        # Loose PDFs directly under misc/ — Omphalina has its own
        # rule above (more specific); this picks up the strays.
        'root':             'misc',
        'kind':             'misc',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        False,
    },
    {
        'root':             'OldBooks',
        'kind':             'book',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        True,
    },
    {
        'root':             'Pholiota',
        'kind':             'book',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        True,
    },
    {
        'root':             'GSMNP',
        'kind':             'book',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        True,
    },
    {
        'root':             'Crepidotus',
        'kind':             'book',
        'container_title':  None,
        'title_from':       'stem',
        'recursive':        True,
    },
]


def _title_after_dash(stem: str) -> str:
    """Strip a ``Prefix-`` from the stem (e.g.
    ``FungaNordica-Russula`` → ``Russula``)."""
    _, _, tail = stem.partition('-')
    return pretty_title_from_stem(tail or stem)


def apply_disk_rule(
    rule: Dict[str, Any], file_relpath: str,
) -> Dict[str, Any]:
    """Build one metadata record from a rule + the PDF's
    site-relative path.  Output shape matches the HTML classifiers
    so downstream code (merge_pdf_records, the backfill helpers)
    needs no special-casing."""
    stem = Path(file_relpath).stem
    record: Dict[str, Any] = {
        'kind':            rule['kind'],
        'container_title': rule['container_title'],
        'source_html':     rule['root'].rstrip('/') + '/',
    }
    title_from = rule['title_from']
    if title_from == 'stem':
        record['title'] = pretty_title_from_stem(stem)
    elif title_from == 'after_dash':
        record['title'] = _title_after_dash(stem)
    elif title_from == 'omphalina':
        parsed = parse_omphalina_filename(stem)
        if parsed is not None:
            record['title'] = parsed['title']
            record['volume'] = parsed['volume']
            record['issue'] = parsed['issue']
        else:
            # Fallback for malformed Omphalina names.
            record['kind'] = 'misc'
            record['title'] = pretty_title_from_stem(stem)
    else:
        record['title'] = pretty_title_from_stem(stem)
    return record


def extract_disk_only_records(
    site_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Walk every ``_DISK_RULES`` directory under ``site_root`` and
    emit ``{pdf_relpath: record}``.  Missing directories are skipped
    silently (lenient per directory)."""
    out: Dict[str, Dict[str, Any]] = {}
    for rule in _DISK_RULES:
        rule_root = site_root / rule['root']
        if not rule_root.is_dir():
            continue
        glob_pattern = '**/*.pdf' if rule['recursive'] else '*.pdf'
        for pdf in rule_root.glob(glob_pattern):
            relpath = str(pdf.relative_to(site_root).as_posix())
            if relpath in out:
                # An earlier (more specific) rule already claimed this
                # file — don't let a broader rule overwrite it.
                continue
            out[relpath] = apply_disk_rule(rule, relpath)
    return out


# ---------------------------------------------------------------------------
# Stage 3 — pipeline + CLI
# ---------------------------------------------------------------------------


def _classifier_for(source_html: str):
    """Pick the per-file classifier by source-HTML filename."""
    name = source_html.rsplit('/', 1)[-1]
    if name == 'journals.html':
        return classify_journals_row
    if name == 'literature.html':
        return classify_literature_row
    if name == 'keys.html':
        return classify_keys_row
    if name in ('miscellanea.html', 'references.html'):
        return classify_misc_row
    return None  # systematics.html (index), index.html, etc.


def _resolve_pdf_relpath(href: str, source_html: str) -> str:
    """Resolve a row's href to a path relative to the site root.

    ``source_html`` is the path of the HTML file the link came from
    (relative to site root).  Hrefs in the HTML are typically
    relative to that file's directory.  ``..`` segments are
    collapsed so the result points at the canonical location of the
    PDF on disk (a few mykoweb rows reference files outside the
    systematics/ subtree via ``systematics/../CAF/...``)."""
    import posixpath
    src_dir = Path(source_html).parent
    joined = (src_dir / href).as_posix()
    return posixpath.normpath(joined)


def _site_html_files(site_root: Path) -> List[Path]:
    """All ``*.html`` files under the systematics tree (no .orig)."""
    sys_dir = site_root / 'systematics'
    files: List[Path] = []
    if (site_root / 'systematics.html').exists():
        files.append(site_root / 'systematics.html')
    if sys_dir.is_dir():
        files.extend(sorted(sys_dir.glob('*.html')))
    return files


def run(
    site_root: Path,
    out_path: Path,
    unparsed_out_path: Optional[Path],
    verbosity: int = 1,
) -> int:
    """Walk the site, classify every PDF row, write JSON outputs.

    Returns the number of rows written to ``out_path``.
    """
    metadata: Dict[str, Dict[str, Any]] = {}
    unparsed: List[Dict[str, Any]] = []
    files = _site_html_files(site_root)
    if verbosity >= 1:
        print(f'Scanning {len(files)} HTML files under {site_root}')

    for html_path in files:
        source_html = str(html_path.relative_to(site_root).as_posix())
        classifier = _classifier_for(source_html)
        if classifier is None:
            if verbosity >= 2:
                print(f'  skip {source_html} (no classifier)')
            continue
        html = html_path.read_text(encoding='utf-8', errors='replace')
        rows = extract_rows_from_html(html, source_html)
        if verbosity >= 1:
            print(f'  {source_html}: {len(rows)} PDF rows')
        for row in rows:
            pdf_relpath = _resolve_pdf_relpath(row['href'], source_html)
            record = classifier(row)
            if pdf_relpath in metadata:
                # Two rows resolved to the same PDF (e.g. the
                # Melanogaster row's split anchors, or a PDF
                # referenced from two HTML files).  Merge instead
                # of overwriting so the better fields survive.
                record = merge_pdf_records(metadata[pdf_relpath], record)
            metadata[pdf_relpath] = record
            if (record.get('kind') == 'book'
                    and row.get('li_text_after', '').strip()):
                # Book fallback but the tail had *something* — surface
                # for review (operator may want to hand-fix, or run an
                # LLM postprocess).
                unparsed.append({
                    'pdf_relpath':   pdf_relpath,
                    'anchor_text':   row['anchor_text'],
                    'li_text_after': row['li_text_after'],
                    'source_html':   source_html,
                })

    # Second pass: walk disk-only directories (CAF, misc/Omphalina,
    # OldBooks, etc.).  PDFs already classified from HTML win — only
    # add records that the HTML pass didn't see.
    disk_records = extract_disk_only_records(site_root)
    if verbosity >= 1:
        print(f'  disk-only directories: {len(disk_records)} PDF rows')
    new_from_disk = 0
    for relpath, record in disk_records.items():
        if relpath in metadata:
            # An HTML row already covered this PDF — prefer the
            # HTML-derived record (it has a citation, an author,
            # a year — strictly more information than the
            # filename-only disk record).
            continue
        metadata[relpath] = record
        new_from_disk += 1
    if verbosity >= 1:
        print(f'  disk-only added {new_from_disk} records '
              f'(others already covered by HTML)')

    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True),
                        encoding='utf-8')
    if verbosity >= 1:
        print(f'Wrote {len(metadata)} rows → {out_path}')
    if unparsed_out_path is not None:
        unparsed_out_path.write_text(
            json.dumps(unparsed, indent=2),
            encoding='utf-8',
        )
        if verbosity >= 1:
            print(f'Wrote {len(unparsed)} unparsed candidate rows → '
                  f'{unparsed_out_path}')
    return len(metadata)


def main() -> int:
    """CLI entry point — parse args, dispatch to ``run()``."""
    parser = argparse.ArgumentParser(
        description='Extract PDF → journal/book metadata from the '
                    'mykoweb systematics tree.',
    )
    parser.add_argument('--site-root', type=Path,
                        default=Path('/data/skol/www/mykoweb.com'),
                        help='Mykoweb site root '
                             '(default: /data/skol/www/mykoweb.com)')
    parser.add_argument('--out', type=Path,
                        default=Path('/data/skol/www/mykoweb.com/'
                                     'systematics_pdf_metadata.json'),
                        help='Output JSON path')
    parser.add_argument('--unparsed-out', type=Path, default=None,
                        help='Optional second JSON listing literature.html '
                             'rows that fell through to the book classifier '
                             'with a non-empty tail (LLM-postprocess input)')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='0=quiet, 1=normal, 2=verbose')
    args = parser.parse_args()

    run(
        site_root=args.site_root.expanduser().resolve(),
        out_path=args.out.expanduser().resolve(),
        unparsed_out_path=(
            args.unparsed_out.expanduser().resolve()
            if args.unparsed_out is not None else None
        ),
        verbosity=args.verbosity,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
