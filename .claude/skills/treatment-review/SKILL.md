---
name: treatment-review
description: Verify an operator's commentary on an extracted SKOL treatment, quantify the defect class corpus-wide, and record it as a pathologies fixture plus a §12.3 memo section. Use when the operator names a taxon_<64 hex> id and describes what is wrong (or right) with it.
---

# Treatment review

The operator reviews treatments in brat or in an HTML dossier and reports
what they see. Your job is **not** to agree. It is to confirm each claim
against the annotation stream, find what they could not see, size the class,
and record it so a detector can be built later.

## 1. Find the treatment

The id in the message is often imperfect — truncated by a character, or
copied from the previous review. Verify before fetching:

```bash
grep -o 'taxon_[0-9a-f]\{64\}' data/annotation_rounds/production_v4_round7.txt | grep '^taxon_<prefix>'
ls data/production_v4/dossiers/production_v4_round7/ | grep '<prefix>'
```

Dossiers live at the repo-relative `data/production_v4/dossiers/<round>/`.
`/home/piggy/src/github.com/piggyatbaqaqi/skol` and
`/data/piggy/src/github.com/piggyatbaqaqi/skol` are the **same directory**
(same inode) — the operator writes the first form, the session's working
directory is the second. Never copy or re-render across them.

To regenerate after a renderer change, pipe ids on stdin and **omit**
`--doc-id`; the flag takes a literal id list, so `--doc-id -` asks for a
treatment named `-` and exits 0 having written nothing:

```bash
grep -o 'taxon_[0-9a-f]\{64\}' \
    data/annotation_rounds/production_v4_round7_review50.txt | sort -u \
  | python bin/treatment_dossier.py --experiment production_v4 \
      --output-dir data/production_v4/dossiers/production_v4_round7
```

If it 404s, the dossier filename is authoritative. If the id resolves but the
content does not match the description, the operator has quoted a neighbouring
id — find the intended one in the round file, **confirm it against every claim
before proceeding**, and say plainly which id you used.

## 2. Fetch the treatment and its annotation stream

```python
import json, urllib.request, base64, os
PW = os.popen("grep -m1 '^COUCHDB_PASSWORD' /home/piggy/.skol_env "
              "| cut -d= -f2- | tr -d '\"'").read().strip()
A = 'Basic ' + base64.b64encode(('admin:' + PW).encode()).decode()

def get(u):
    r = urllib.request.Request('http://localhost:5984/' + u)
    r.add_header('Authorization', A)
    return json.load(urllib.request.urlopen(r))

def raw(u):
    r = urllib.request.Request('http://localhost:5984/' + u)
    r.add_header('Authorization', A)
    return urllib.request.urlopen(r).read().decode('utf-8', 'replace')

t = get('skol_exp_production_v4_02_00_treatments_prose/' + tid)
src = (t.get('ingest') or {})['_id']
ann = raw('skol_exp_production_v4_01_00_ann_combined/%s/article.txt.ann' % src)
paras = ann.split('\n\n')
```

**The paragraph mapping is `index = (paragraph_number - 1) // 2`.** The
treatment's `*_spans` carry `paragraph_number`; the `.ann` is split on blank
lines. Confirm the mapping on the first span before trusting it — it has held
for every document so far but it is a convention, not a guarantee. If it is
off, locate the treatment by searching `paras` for its nomenclature text.

Print the neighbourhood with labels:

```python
import re
for i in range(i0 - 4, i0 + 16):
    p = paras[i]
    lab = sorted(set(re.findall(r'#([A-Za-z-]+)\*\]', p)))
    body = ' | '.join(re.sub(r'\[@|#[A-Za-z-]+\*\]', '', p).split('\n'))
    print('\n[%d] (para %d) %s (%d)' % (i, 2 * i + 1, lab or ['-'], len(body)))
    print('    ', body[:520])
```

## 3. Verify every claim, and look past them

Take each operator claim in turn and confirm or correct it against the
paragraphs. They are usually right; when they are not, say so plainly and
show the evidence. Two things they systematically cannot see:

- **Damage that lands in a neighbouring treatment** (§12.3.63). Check the
  treatment before and after in the same document.
- **Whether the claim generalises.** That is the whole point of the exercise.

Read `docs/data_quality_production_v4_model.md` §12.3 before writing: the
class is usually already named, and the valuable finding is the measurement,
not the anecdote.

## 4. Measure the class

Corpus-wide field scans are one `_find` over ~81 500 treatments:

```python
def post(db, body):
    r = urllib.request.Request('http://localhost:5984/' + db + '/_find',
                               method='POST', data=json.dumps(body).encode())
    r.add_header('Authorization', A)
    r.add_header('Content-Type', 'application/json')
    return json.load(urllib.request.urlopen(r))

docs = post('skol_exp_production_v4_02_00_treatments_prose',
            {'selector': {'treatment': {'$exists': True}},
             'fields': ['_id', 'treatment', 'description', ...],
             'limit': 90000})['docs']
```

Block-level questions need the `.ann` attachments, which are ~7 GB corpus-wide.
Sample instead: `/tmp/annsample` holds a cached deterministic 400-document
sample (`ids[::len(ids)//400]` over `_all_docs`). State the sample size and
the selection rule in the memo.

## 5. Check precision before you publish

**Every over-claim in this series came from skipping this step.** Before any
number goes in the memo:

- **Read the actual hits.** Print 8 random matches and look at them. The
  `Culture characteristics` "schema gap", the `Nova Hedwigia` act count, and
  the 2 100-treatment capture-recapture estimate all died here.
- **Ask whether the variable varies within the confounding unit.** Separator
  style is a *document* constant (126 of 129 documents use exactly one), so
  joined-vs-spaced compares publishers, not blocks — that invalidated
  §12.3.48's causal claim. Prefer a within-document or within-stratum
  comparison.
- **Ask whether the schema could have honoured the signal.** A heading inside
  a field is evidence of a missed boundary only where a label exists for it;
  `biology ← Distribution` is not a segmentation defect because there is no
  `Distribution` tag.
- **Never read a heading as the label.** §12.3.59: a `Diagnosis` field is more
  often headed `Notes` than `Diagnosis`, and the classifier is right to
  disagree with the paper. Headings locate sections; they do not name them.

If a measurement dies, record it in the memo as a worked example of why —
that is more useful than silence.

## 6. Compute the detector flags

```python
from treatments_to_structured.triage_signals import (
    predicted_issues, treatment_signals)
from treatments_to_structured.merge_metric import treatment_merge_metric
from treatments_to_structured.gn_client import authored_binomial_in_text

abd = authored_binomial_in_text(t.get('description') or '')
tr = {'description': t.get('description') or '',
      'diagnosis': t.get('diagnosis') or '',
      'synthetic_nomenclature': bool(t.get('synthetic_nomenclature')),
      'description_spans': t.get('description_spans') or [],
      'source_anchors': t.get('source_anchors') or []}
flags = predicted_issues(
    treatment_signals(tr, authored_binomial_in_desc=abd),
    treatment_merge_metric(tr))
```

Run under `conda activate skol3.14` with `PYTHONPATH=.`.

## 7. Record it

**Fixture** — `tests/fixtures/pathologies.json`, appended to `pathologies`
(`poster_children` is for treatments with *no* defect). Required fields, or
`TestFixtureIntegrity` fails: `id`, `source_experiment`, `source_db`,
`captured_at`, `captured_rev`, `class`, `description`, `diagnosis`,
`synthetic_nomenclature`, `authored_binomial_in_desc`, `expected_flags`,
`known_missed_flags`, `notes`.

- `notes` is the **narrative** field. A treatment's own notes text goes in
  `treatment_notes`.
- `expected_flags` must be what `predicted_issues` actually returns — it is a
  regression guard, not a wish.
- `known_missed_flags` is where the new detector proposal is named.
- Add `source_anchors: []` **only** when the treatment genuinely has none;
  the harness substitutes a placeholder otherwise, by design.
- Every taxon named in the memo needs its own entry, including a neighbour
  cited as a victim.

```bash
conda activate skol3.14 && PYTHONPATH=. python -m pytest -q tests/pathologies_test.py
```

`pytest tests/` alone hits `INTERNALERROR: SystemExit: 0`; name the file.

**Memo** — a new `### 12.3.N` section at the top of the §12.3 run in
`docs/data_quality_production_v4_model.md`, inserted immediately before the
current highest-numbered section (they are in descending order). Quote the
operator verbatim, show the paragraph table, give the measurement, and name
the fixture and its flags at the end. When a finding corrects an earlier
section, **annotate that section in place** with a `> **CORRECTED by
§12.3.N.**` blockquote — leave a blank line after the blockquote or it will
swallow the next paragraph.

**Commit** the memo and fixture together, with the numbers in the message.

## 8. Report the disposition

Per `CLAUDE.md`: say what was recorded (class and identifier), where (file and
section), and the commit. Then name anything you chose **not** to record and
why — a measurement you discarded, a class you filed as a cross-reference to
an existing section rather than a new one, code you did not change. A hash
alone is not a disposition.
