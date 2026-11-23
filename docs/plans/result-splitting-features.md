# Result-splitting feature tools

Status: F1 unparked 2026-11-23, tests drafted for approval
(`treatments_classifier/result_splits_test.py`).  F2 and F3 stay
parked behind the JSON representation work
(`docs/plans/vocab-tree-narrowing.md`, Proposals 2 and 3).  F1 takes
keys as input, so it needs only P2a's synonym map, which is on
production; later canonicalisation flows through it unchanged.

The Text Features and JSON Features tabs are meant to suggest what to
add to a description next, to narrow the search.  Today they rank
features by decision-tree importance, and the ranking is degenerate.
This plan adds two new tools that measure something direct: how a
feature splits the current search results.  The JSON tool follows the
anatomy, so the user can see which part, and which attribute of it,
separates the results.

## Principle: a radical change of behaviour is a new tool

Decided by the user, 2026-11-18.  Refining a tool is fine; a radical
change of what a tool does is a new tool, and the old one stays.
There will eventually be dozens of tools (for example, templates of
what to record for a group of organisms, such as agarics), and the
current tab layout may not last.  Which tools are made most visible
will be decided by how much each is used, which needs usage tracking
in the UI (see "Usage tracking" below).

So this plan adds two tools beside Text Features and JSON Features
rather than replacing them.

## What is wrong today

`TreatmentsDecisionTreeClassifier` and `TreatmentsJSONClassifier`
(`treatments_classifier/`) train a scikit-learn decision tree on the
search results, one class per result
(`treatments_json_classifier.py:369`).  The widget shows each feature's
importance as a percentage of the top feature's
(`FeatureSelectionWidget.jsx:121`).

With one result per class, the Gini criterion lowers impurity by
exactly 1/N at every split, whether the split is 100/100 or 199/1.  A
feature's importance is then just the number of times the tree used
it.  A feature used once scores the same as every other feature used
once, so every row reads 100%.  The tree's choice of split is close to
arbitrary for the same reason.  On 200 synthetic documents, the
current Gini tree's root split was 177/23; an entropy tree's was
102/98.

Three smaller problems:

* The JSON tokens are `path=value`, with the path joined by `_` (which
  also appears in key names) and list indices kept
  (`Ascomata_Color_0=Brown`, `Ascomata_Color_1=Black`), so one concept
  becomes several features.
* Only leaf values are features.  Nothing says "this part, or this
  attribute, is where the results differ".
* The tree the endpoints build (`tree_json`) is returned and never
  used by the frontend.

### Finding: "Taxa for Features" is a dead setting

`UserSettings.feature_taxa_count` (Settings label "Taxa for Features",
default 6, range 2–50) is saved, serialized and exported
(`search/export_service.py:69`), but nothing reads it.  The widget
builds its features from every result of the search, which is the
`k` setting (default 20, maximum 100), and shows at most
`feature_top_n` of them (label "Feature Count", default 30).  Removing
the dead setting is a refinement of the existing tools, separate from
this plan.

## Which results are counted

N is the number of results of the search: all of them, which is `k`
(up to 100).  The number of rows shown is a separate cap,
`feature_top_n`.  Every ratio below is against N, never against the
number of rows shown.

## The measure: how much a feature separates the results

A question has answers, and each answer picks out a group of the
current results.  If answer *a* covers *n_a* of the N results, the
other `1 - n_a/N` of them differ from it.  Assuming the user's fungus
is like a random result, answer *a* comes up with probability
proportional to *n_a*.  The score is the expected share of results
that differ from the user's answer:

    score = Σ_a  p_a · (1 − n_a / N),    p_a = n_a / Σ_b n_b

**A single feature** has two answers, "yes" (in k results) and "no"
(N − k).  Its score is `2k(N − k)/N²`, at most 50%, when the feature
splits the results in half.  A feature every result shares scores 0,
which is right: it is probably already in the user's description.

**An attribute** (a JSON leaf key such as `Pileus / Color`) has one
answer per value, plus "not mentioned" for the results that do not
state it.  It can score above 50%, which is the point: a many-valued
attribute can carry more information than any one feature.  The value
rows also show each answer on its own:

| Pileus / Color — separates about 60% | In | If yours is this, the others |
|---|---|---|
| Brown | 100 of 200 | 50% |
| Yellow | 80 of 200 | 60% |
| Blue | 10 of 200 | 95% |
| (not mentioned) | 20 of 200 | |

So the attribute row says "colour matters here", and the Blue row
says "if yours is blue, most of these results are not like it".
(The 60% is the score above, computed with the four answers: 0.598.)

Results that list several values (`Color: [Brown, Black]`) count
under each.  So the answer counts can sum to more than N, and the
weights `p_a` use their sum.

### The measure describes the results, not the next search

The search is SBERT embedding similarity: it embeds the whole
description and ranks results against it.  Adding "pileus blue" moves
the query, so results describing a blue pileus rise.  Results that
describe a brown pileus, or none, are re-ranked, not removed, and how
far they fall depends on the embedding.  So the tools cannot promise
that a feature "rules out" a share of the results, and they do not
say so.  The score describes how the feature divides the current
results, which is useful on its own: it says where those results
disagree.

That keeps "not mentioned" simple.  It is one more group of the
current results, scored like the others and shown with its count, so
a split that is really about how thorough the descriptions are is
visible.  Nothing in the search changes.

A tool that actually filters (keep only the results whose JSON says
`Pileus / Color = Blue`, with a choice of keeping or dropping those
that do not mention it) would change the core search behaviour.  By
the principle above it would be a new tool with its own plan, not part
of this one.

## The JSON tool follows the anatomy

Each result's `json_annotated` becomes a set of features at every
level of its key path, with list indices dropped and keys kept as
written:

* `Ascomata` — the part is described;
* `Ascomata / Peridium` — a nested part;
* `Ascomata / Color` — an attribute (a key whose value is a list of
  strings);
* `Ascomata / Color = Brown` — a value.

The tool shows a collapsible list grouped by part (the numbers are
illustrative, apart from the colour row worked above):

    Ascomata                described in 180 of 200
      Color                 separates ~60%
        Brown   100 · Yellow 80 · Blue 10 · (not mentioned) 20
      Peridium
        Texture             separates ~31%
          ...
    Asci                    described in 200 of 200
      Ring                  separates ~48%
        ...

Parts are ordered by their best attribute's score and attributes by
their own score, so whichever level of the anatomy actually separates
the results comes first.  A part every result describes, with
attributes every result shares, sinks to the bottom.  The ranking
does not have to walk the anatomy top-down: a strict rule (only ask
about children of the part last chosen) would sometimes force a
useless question before a useful one.

A part row's "described in k of N" is itself a feature, and counts as
one when ordering parts.

Ticking a value adds `Ascomata color brown;` to the description, as
the JSON Features tab's `display_text` does.

## The text tool

There is no hierarchy in prose, so the text tool is a flat list: word
n-grams (1–2, lowercased, accents stripped, as today) from each
result's description, scored as single features and labelled
"in k of N".  When several n-grams occur in exactly the same results
("clamp", "clamp connections"), the list shows one row, the longest
(decided 2026-11-18).

## Steps

Built alongside the existing tools; nothing is removed.

1. **F1, scoring (pure Python).** A new module,
   `treatments_classifier/result_splits.py`, with the JSON feature
   extraction, the text n-grams and the score, with `*_test.py`.  No
   Django, no CouchDB.
2. **F2, endpoints.** New `POST /api/features/json/` and
   `POST /api/features/text/`, fetching the result documents as the
   old views do (same experiment database resolution) and returning
   the structures below.
3. **F3, widget.** Two new tabs with their panels, and Jest tests for
   both.  Text Features and JSON Features stay as they are.

Whether the old tabs are later retired is a decision for the usage
data, not this plan.

## Response shapes (F2)

JSON:

```json
{
  "n_results": 200,
  "parts": [
    {
      "path": ["Ascomata"],
      "described": 180,
      "score": 0.598,
      "attributes": [
        {
          "path": ["Ascomata", "Color"],
          "score": 0.598,
          "not_mentioned": 20,
          "values": [
            {"value": "Brown", "count": 100, "others": 0.5,
             "display_text": "Ascomata color brown"}
          ]
        }
      ],
      "parts": []
    }
  ]
}
```

Text:

```json
{
  "n_results": 200,
  "features": [
    {"text": "clamp connections", "count": 98, "score": 0.4998}
  ]
}
```

`feature_top_n` caps the number of attributes in the JSON response
and features in the text one.

## Usage tracking and UI audit

Needed before deciding which tools to make most visible, or whether
to retire any.  Not designed here; it needs its own plan (Trello list
"Description tools"), covering at least: which events count as using
a tool (opening its tab, adding a feature from it to the description,
a search that follows), whether signed-out use is counted, where the
counts live, and what is kept about the user.

The same plan covers a broader UI audit (user, 2026-11-18), answering:

* Which parts of the UI are used most?
* Which UI features does nobody use?
* Can mistakes in using the site be detected, the kind that call for
  a redesign?  (For example, an action undone straight away, a
  setting changed and changed back, a form that fails validation
  repeatedly, or a search repeated with no change.)

The two new tools should record usage from their first release if
that plan lands first.

## Sequencing

1. The JSON representation work in `docs/plans/vocab-tree-narrowing.md`
   (the `data/` move, P2a–P2c, Proposal 3), to completion.
2. This plan, F1–F3.  The synonym map from P2 then applies from the
   start, so `Ascospore` and `Ascospores` are one part.

## Open questions

1. **Names.**  Decided 2026-11-23: "Text Splits" and "Structured
   Splits".  "JSON" names an internal data structure, not what the
   user sees; "Structured" matches "Text description" and "Structured
   description".  Other suggestions are still welcome.  This plan
   says "the JSON tool" below for the Structured Splits tool.

## Decisions the F1 tests make (for approval, 2026-11-23)

The plan left these open; the tests in
`treatments_classifier/result_splits_test.py` pin them.

1. **Values compare without case.**  `Brown` and `brown` are one
   value, shown in its most common written form.  Keys stay as
   written.
2. **A part's score is its best feature**: the highest of its own
   "described" score, its attributes' scores, and its nested parts'
   scores.  That is what lets a nested part lift its parent.
3. **`top_n` caps attributes across the whole tree**, best first.  A
   part left with no attribute beneath it is dropped.
4. **`display_text`** is the first key as written, then the rest of
   the path and the value lowercased: `Ascomata color brown`.
5. **Ties** break on count, then on path or text, alphabetically, so
   the order is stable.
6. **Text tokens** are words of two or more word characters, as
   today's TF-IDF tokenizer has them, and word pairs may cross
   punctuation, also as today.
7. **"The longest" n-gram** is the one with most words, then most
   characters, then first alphabetically.
8. **A result with no structured description still counts in N.**
9. **Not yet decided:** booleans and numbers as values.  The tests
   cover strings only.

## Trello (list "Description tools", 2026-11-18)

* F1, scoring module: https://trello.com/c/Wjr7FsGq
* F2, endpoints: https://trello.com/c/3MPvW6u4
* F3, widget tabs: https://trello.com/c/GGrzZ79l
* Design, usage tracking and UI audit: https://trello.com/c/VwB64E1c
* Remove the dead "Taxa for Features" setting: https://trello.com/c/XLZmBdno
