# Approved content licences

`data/approved_licenses.json` is the machine-readable list of content
licences SKOL may ingest; `ingestors/licenses.py` reads it (through
`skol_share`, so it ships in the deb's `share/`).  This page records the
rule behind it.

## The rule (operator, 2026-11-23)

SKOL is a non-commercial, permanently free-access GPL-3.0 project
(`docs/structure_ocr.md` §1).  For content:

| Licence | Approved | Why |
|---|---|---|
| Public domain (BHL `NOT_IN_COPYRIGHT`, Public Domain Mark) | yes | no restriction |
| CC0 | yes | no restriction |
| CC BY, CC BY-SA (any version) | yes | open |
| CC BY-NC, CC BY-NC-SA (any version) | yes | SKOL is non-commercial |
| CC BY-ND, CC BY-NC-ND (any version) | **no** | SKOL derives text (extraction, JSON) |
| "No known copyright restrictions" | **no** | not positive evidence; BHL itself says "proceed with caution" |
| Blank, "Not specified", "Other" | **no** | not positive evidence, although BHL counts them as public domain |
| Anything unrecognised | **no** | recorded on the document so it can be added |

**Public domain by date** (operator, 2026-11-23): a work *published*
before the current year minus 95 (the US term for published works) is
approved as `public-domain-by-date` when its stated rights are blank,
"Not specified"/"Other", "No known copyright restrictions", or not
recognised.  A positive licence (e.g. CC BY-ND) or an explicit
in-copyright statement keeps its own verdict.  The year is a volume's
latest year (BHL `EndYear`, else `Year`) or an article's own date.
Author death dates are not looked up, so unpublished works get no
date rule beyond this.  The number lives in `approved_licenses.json`
(`public_domain_by_date.years`).

Share-alike licences bind what SKOL publishes from those texts to the
same licence.

The component rule is separate and stricter: software and models must
be OSI-approved, and non-commercial model licences are rejected
(`docs/plans/production-v5-execution.md`, "Model candidates are
licence-constrained", 2026-09-14).

## Identity

Each licence has a stable id: its SPDX identifier where one exists
(`CC0-1.0`, `CC-BY-NC-SA-4.0`, ...), otherwise a fixed id
(`public-domain`, `no-known-copyright`, `unspecified`).  Documents store
the id (`license_id`) and the raw statement (`license_raw`); raw
statements are matched, never used as keys.  Creative Commons ids are
derived from the licence URL's family and version, so every version is
covered without listing it.

## Enforcement

Only the BHL ingestor (`ingestors/bhl.py`) enforces the list so far.
Only approved content may be *processed*; storing a complete
mixed-licence volume locally is accepted (operator, 2026-11-23).  So
for a volume whose articles have mixed rights:

* `article.pdf`, the attachment every pipeline step and the web site
  read, keeps a page only if every article touching it is approved
  (pages outside any article follow the volume's rights); other pages
  are blank pages of the same size, so PDF page numbers still line up.
* The complete download is kept outside CouchDB, at its mirror path
  under `/data/skol/www` (`SKOL_WWW_DIR`; `ingestors/web_mirror.py`),
  and recorded as `complete_volume` (`path`, `sha256`, `bytes`).  It
  must not be stored as a CouchDB attachment: `PDFAttachmentView`
  serves any attachment by name to anyone.
* Articles are located by `pdf_page_ranges` (runs of PDF pages; an
  article's plates are often bound elsewhere in the volume).
