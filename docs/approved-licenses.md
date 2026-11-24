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
For a volume whose articles have mixed rights, it keeps a page only if
every article touching it is approved (pages outside any article follow
the volume's rights) and replaces the rest with blank pages, so PDF page
numbers still line up.
