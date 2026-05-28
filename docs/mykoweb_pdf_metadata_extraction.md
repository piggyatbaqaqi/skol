# Mykoweb PDF → journal / book metadata extraction

Status: **planned**.  Implement when picking up the mykoweb ingestor.

## Goal

Build a `{pdf_relpath: {kind, title, container_title, ...}}` mapping
covering every PDF linked from
`/data/skol/www/mykoweb.com/systematics.html` and its subdirectories.
The mapping is consumed later by the mykoweb ingestor so it can tag
each treatment with its source journal / book.

Output: flat JSON file at
`/data/skol/www/mykoweb.com/systematics_pdf_metadata.json` (path
relative to skol_dev; pick the final landing spot at integration
time).  No DB writes from this script.

## Source-tree shape

| HTML file | PDFs | Citation shape |
|---|---|---|
| `systematics/journals.html` | 258 | `<li>JOURNAL <a href="journals/JOURNAL/...pdf">Vol. N No. M</a></li>` — journal name is **literally the parent directory** of the PDF path |
| `systematics/literature.html` | 222 | `<li><strong>AUTHOR</strong> (YEAR). <a href="literature/...pdf">TITLE</a> CONTAINER-CITATION.</li>` — full bibliographic citation, container appears in text after `</a>` |
| `systematics/keys.html` | 22 | Taxonomic keys, mostly off-site |
| `systematics/miscellanea.html` | 4 | Mixed tools/lists |
| `systematics/references.html` | 3 | Mixed |
| `systematics.html` | 0 | Index page only |

509 PDF links total; 480 of them (journals + literature) are highly
regular and yield to pure parsing.

## Approach

Pure-programmatic with BeautifulSoup.  No LLM in the hot path.

### Stage 1 — extract raw rows

For every `<a href>` ending in `.pdf` in each HTML file under
`/data/skol/www/mykoweb.com/systematics/`, capture:

- `source_html` — which file the link came from
- `href` — resolved to a path relative to `mykoweb.com/`
- `anchor_text` — visible text of the `<a>`
- `li_text_before` — text inside the enclosing `<li>` before the `<a>`
- `li_text_after` — text inside the enclosing `<li>` after the `</a>`
- `strong_text` — text of any `<strong>` / `<b>` in the `<li>` (author)
- `year_match` — `(YYYY)` near the start of the `<li>`, if present

The raw rows are useful as a cache so the classifier stage can be
re-run without re-parsing HTML.

### Stage 2 — classify per source file

| Source file | Rule |
|---|---|
| `journals.html` | `kind = "journal"`; `container_title = href.parts[-2]` (parent directory of PDF); `volume_info = anchor_text`; reliable 100% |
| `literature.html` | Try to parse `li_text_after` with `(?P<container>.+?)\s+(?P<vol>\d+)(?:\((?P<issue>\d+)\))?:\s*(?P<pages>[\d\-–]+)` → `kind = "journal_article"`, fields populated.  No match → `kind = "book"`, `title = anchor_text`, `container_title = None`.  `title`, `author`, `year` come from anchor / strong / year_match. |
| `keys.html` | `kind = "key"`; just record `title = anchor_text`.  Not bibliographic, no container_title. |
| `miscellanea.html` / `references.html` | `kind = "misc"`; record anchor_text + li_text_after verbatim.  Small enough to hand-fix in postprocess if needed. |

### Stage 3 — emit JSON

```json
{
  "systematics/journals/Mycotaxon/Mycotaxon v001n1.pdf": {
    "kind": "journal",
    "title": "Mycotaxon Vol. 1 No. 1",
    "container_title": "Mycotaxon",
    "volume_info": "Vol. 1 No. 1",
    "source_html": "systematics/journals.html"
  },
  "systematics/literature/The North American Species of Clavaria.pdf": {
    "kind": "journal_article",
    "title": "The North American Species of Clavaria with Illustrations of the Type Specimens",
    "container_title": "Annals of the Missouri Botanical Garden",
    "volume": "9",
    "issue": "1",
    "pages": "1-78",
    "author": "Burt, E.A.",
    "year": 1922,
    "source_html": "systematics/literature.html"
  },
  "systematics/literature/A Monograph of Favolaschia.pdf": {
    "kind": "book",
    "title": "A Monograph of Favolaschia",
    "container_title": null,
    "source_html": "systematics/literature.html"
  }
}
```

Also emit a sibling `*.unparsed.json` listing every literature.html
row whose `li_text_after` didn't match the citation regex — that's
the candidate set for hand-fix or LLM postprocess.

## Where to put the code

- `bin/extract_mykoweb_pdf_metadata.py` — the extractor (CLI:
  `--site-root`, `--out`, `--unparsed-out`, `--verbosity`)
- `bin/extract_mykoweb_pdf_metadata_test.py` — pytest module per
  CLAUDE.md.  Cover: journals.html-shape row, literature.html row
  with full citation tail, literature.html row with no tail (book),
  HTML-entity unescaping (`&#32;` in hrefs), off-site `<a>` skipped,
  `<strong>` / `<b>` interchangeable for author, year regex misses
  cleanly (no crash).

No Redis key, no CouchDB writes — this script just produces a JSON
artifact.  The mykoweb ingestor will consume it later.

## LLM fallback (only if needed)

If `*.unparsed.json` is non-trivial (>20 rows), wrap a single
Anthropic Messages API call that takes the unparsed citation tails
in a batch and returns structured `{container_title, volume,
pages, ...}` per row.  Cost is bounded — 222 rows ceiling, almost
certainly far less.  Skip entirely if the regex handles it.

## Integration with mykoweb ingestor (out of scope here)

The ingestor reads the JSON and uses `container_title` as the
`journal` field on each ingested treatment doc in skol_dev,
mirroring how `backfill_journal.py` populates that field from
Crossref for DOI-bearing docs.  Books will need a separate
`book_title` field or a `kind == "book"` flag — decide at
integration time.

## Notes / caveats

- HTML entities in hrefs (`&#32;` for space) must be unescaped
  before the path is used to look up files on disk.
- `journals.html` is the cleanest source — the parent-directory
  rule is more reliable than the leading `<li>` text (some rows
  have whitespace / formatting noise).
- Off-site `<a href="http(s)://...">` links exist in `keys.html`
  and `references.html`; skip these (no local PDF to map).
- The `.orig` files next to each HTML are pre-edit snapshots;
  ignore them — parse only the active `.html` files.
