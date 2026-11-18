# Replacing the Text and JSON Feature tools

Status: design, awaiting review (2026-11-18).

The Text Features and JSON Features tabs are meant to suggest what to
add to a description next, to narrow the search.  Today they rank
features by decision-tree importance, and the ranking is degenerate.
This plan replaces it with a direct measure: how much of the current
result set each feature would rule out.  It also makes the JSON tab
follow the anatomy, so the user can see which part, and which
attribute of it, separates the results.

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

## The measure: expected share of results ruled out

The search loop already conditions the questions: the user adds a
feature, searches again, and gets a new list ranked against the
narrowed results.  So a feature only needs to be scored against the
current N results, with no tree.

A question has answers, and each answer keeps some of the results.  If
answer *a* keeps *n_a* of the N results, it rules out `1 - n_a/N`.
Assuming the user's fungus is like a random result, answer *a* comes
up with probability proportional to *n_a*.  The score is the expected
share ruled out:

    score = Σ_a  p_a · (1 − n_a / N),    p_a = n_a / Σ_b n_b

**A single feature** has two answers, "yes" (in k results) and "no"
(N − k).  Its score is `2k(N − k)/N²`, at most 50%, when the feature
splits the results in half.  A feature every result shares scores 0,
which is right: it is probably already in the user's description.

**An attribute** (a JSON leaf key such as `Pileus / Color`) has one
answer per value, plus "none of these" for the results that do not
mention it.  It can score above 50%, which is the point: a
many-valued attribute can carry more information than any one
feature.  The value rows also show what each answer would do on its
own:

| Pileus / Color — rules out about 60% | In | If yours is this, rules out |
|---|---|---|
| Brown | 100 of 200 | 50% |
| Yellow | 80 of 200 | 60% |
| Blue | 10 of 200 | 95% |
| (not mentioned) | 20 of 200 | |

So the attribute row says "colour matters here", and the Blue row
says "if yours is blue, that settles most of it".  (The 60% is the
score above, computed with the four answers: 0.598.)

Results that list several values (`Color: [Brown, Black]`) count
under each.  So the answer counts can sum to more than N, and the
weights `p_a` use their sum.

### What absence means

In LLM-extracted JSON, an attribute missing from a result usually
means "not described", not "absent".  This design treats "not
mentioned" as an answer that rules out the results which do mention
the attribute.  That matches what the next search does: the search is
embedding similarity, and results that do not mention the added term
fall.  The "(not mentioned)" count is shown, so a split that is really
about how thorough the descriptions are is visible.  This is open
question 2.

## The JSON tab follows the anatomy

Each result's `json_annotated` becomes a set of features at every
level of its key path, with list indices dropped and keys kept as
written:

* `Ascomata` — the part is described;
* `Ascomata / Peridium` — a nested part;
* `Ascomata / Color` — an attribute (a key whose value is a list of
  strings);
* `Ascomata / Color = Brown` — a value.

The tab shows a collapsible list grouped by part (the numbers are
illustrative, apart from the colour row worked above):

    Ascomata                described in 180 of 200
      Color                 rules out ~60%
        Brown   100 · Yellow 80 · Blue 10 · (not mentioned) 20
      Peridium
        Texture             rules out ~31%
          ...
    Asci                    described in 200 of 200
      Ring                  rules out ~48%
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
today's `display_text` does.

## The Text tab

There is no hierarchy in prose, so the Text tab stays a flat list:
word n-grams (1–2, lowercased, accents stripped, as today) from each
result's description, scored as single features and labelled
"in k of N".  Two n-grams that occur in exactly the same results are
one row (open question 3).

## New tools or replacements

### Two new tools, keeping the old ones

Pros:

* No regression risk: the old tabs keep working while the new ones
  are checked on real searches.
* Side-by-side comparison is possible.
* The old endpoints, their tests (`search/tests/test_feature_views.py`)
  and their settings are untouched until a later decision.

Cons:

* Six tabs: Vocabulary, Text Features, JSON Features, the two new
  ones, and Metrics.  The user has to know which to trust.
* The old tabs show numbers that are known to be wrong.  The
  side-by-side comparison has little to compare against: the old
  ranking is arbitrary.
* Two sets of settings: the old tabs need `max_depth`, `min_df` and
  `max_df`; the new ones do not.
* Two migrations: the old tabs would still have to go later, with the
  same removal work, plus the work of having kept them.

### Replacing the old ones

Pros:

* Four tabs, as today, with numbers that mean what they say.
* The Settings menu loses three fields that only made sense for the
  tree (`feature_max_tree_depth`, `feature_min_df`, `feature_max_df`).
* The `treatments_classifier` package (about 1,700 lines) and its
  per-request model training leave the request path; counting
  features is much cheaper than fitting a tree.  scikit-learn stays a
  dependency: `finder.py` and `skol/constrained_decoder.py` use it.

Cons:

* The REST responses change shape, so `docs/api-reference.md`, the
  view tests and the widget change together.
* Removing `treatments_classifier` is a packaging change (pyproject
  and the deb).
* No way back short of reverting.

### Recommendation

Replace them, but build the new code alongside and remove the old
code last, so every commit keeps the head green and each step can be
hand-checked:

1. **F1, scoring (pure Python).** A new module with the JSON feature
   extraction, the text n-grams and the score, with `*_test.py`.  No
   Django, no CouchDB.
2. **F2, endpoints.** New `POST /api/features/json/` and
   `POST /api/features/text/`, fetching the result documents as the
   old views do (same experiment database resolution) and returning
   the grouped structure below.
3. **F3, widget.** The Text Features and JSON Features tabs switch to
   the new endpoints and panels.  Tab names stay.  Jest tests for both
   panels.
4. **F4, removal.** The old endpoints and their tests, the
   `treatments_classifier` package (pyproject, deb), the three
   settings fields (with a migration), and the docs.

The old tabs are never shown next to the new ones.

## Response shape (F2)

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
            {"value": "Brown", "count": 100, "rules_out": 0.5,
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

`top_n` (the existing `feature_top_n` setting) caps the number of
attributes in the JSON response and features in the text one.

## Open questions

1. **How many results.**  The feature lists are built from the top
   `feature_taxa_count` results, default 6 (range 2–50).  With 6, a
   split reads "3 of 6" and the scores are coarse.  Should the default
   rise, for example to 50?  The cost is fetching more documents per
   request.
2. **Absence.**  Treat "not mentioned" as an answer that rules out the
   mentioning results (proposed above), or as unknown, keeping them?
   Keeping them is safer in principle, but then a value that results
   only ever state positively ("Fasciculation: Present") can never
   rule anything out.
3. **Text duplicates.**  When n-grams cover the same results
   ("clamp", "clamp connections"), show the longest or the shortest?
   Proposed: the longest, as it reads as a phrase.
4. **Synonyms.**  Vocab-tree P2 (`docs/plans/vocab-tree-narrowing.md`)
   folds JSON synonyms ("Ascospore" / "Ascospores").  The JSON tab
   would benefit from the same map once it exists.  Proposed: not a
   blocker; apply the map when P2 lands.
