"""Score how features split a set of search results (plan F1).

Pure Python: no Django, no CouchDB.  See
``docs/plans/result-splitting-features.md``.

A question has answers, and answer *a* picks out ``n_a`` of the ``N``
current results.  The score is the expected share of results that
differ from the user's answer::

    score = sum_a p_a * (1 - n_a / N),    p_a = n_a / sum_b n_b

Two tools use it:

* **Structured Splits** walks each result's structured description
  (a JSON object) and scores parts, attributes and values.
* **Text Splits** scores word 1- and 2-grams of each result's text.

Decisions the tests pin are listed in the plan, "Decisions the F1
tests make".
"""

import re
import unicodedata
from collections import Counter, defaultdict
from typing import (Any, DefaultDict, Dict, FrozenSet, List, Mapping,
                    Optional, Sequence, Set, Tuple)

from treatments_to_structured.vocab_canonical import value_group

Path = Tuple[str, ...]


TOKEN_RE = re.compile(r'(?u)\b\w\w+\b')


def split_score(counts: Sequence[int], n_results: int) -> float:
    """Return the expected share of ``n_results`` that differ from an
    answer, given each answer's count of results.

    Counts may sum to more than ``n_results`` (a result can give
    several answers).  Zero results, or all-zero counts, score 0.
    """
    total = sum(counts)
    if n_results <= 0 or total <= 0:
        return 0.0
    return sum((c / total) * (1 - c / n_results) for c in counts)


def feature_score(count: int, n_results: int) -> float:
    """Score a single yes/no feature present in ``count`` results:
    ``2k(N - k) / N**2``."""
    if n_results <= 0:
        return 0.0
    return 2 * count * (n_results - count) / n_results ** 2


def structured_features(
        doc: Mapping[str, Any]) -> Tuple[FrozenSet[Path],
                                         Dict[Path, Set[str]]]:
    """Return ``(parts, attributes)`` for one structured description.

    * ``parts``: every key path whose value is an object (or a list
      holding objects), list indices dropped, keys as written.
    * ``attributes``: every key path whose value is a string (or a list
      holding strings), mapped to the set of its values as written.

    A key can be both a part and an attribute when its list mixes
    objects and strings.  Empty strings and ``None`` are ignored.
    Booleans and numbers are kept as their ``str()`` for now (plan,
    decision 9, open).
    """
    parts: Set[Path] = set()
    attributes: Dict[Path, Set[str]] = {}

    def walk(path: Path, value: Any) -> None:
        if isinstance(value, Mapping):
            if path:
                parts.add(path)
            for key, child in value.items():
                walk(path + (str(key),), child)
        elif isinstance(value, list):
            for item in value:
                walk(path, item)
        elif value is not None and path:
            text = str(value).strip()
            if text:
                attributes.setdefault(path, set()).add(text)

    walk((), doc)
    return frozenset(parts), attributes


def _owner(path: Path) -> Path:
    """The part an attribute is listed under: its parent, or itself
    for a top-level attribute."""
    return path[:-1] if len(path) > 1 else path


def _display_text(path: Path, value: str) -> str:
    return ' '.join([path[0]] + [p.lower() for p in path[1:]]
                    + [value.lower()])


def _most_common(forms: "Counter[str]") -> str:
    """The most frequent written form, ties to the first
    alphabetically."""
    return min(forms, key=lambda f: (-forms[f], f))


def _score_attribute(path: Path, docs_values: List[Set[str]],
                     n_results: int) -> Dict[str, Any]:
    """Score one attribute from each mentioning result's values.

    Values are one when ``value_group`` says so (decision B), and are
    shown in their most common written form; ``present`` and
    ``absent`` show as themselves, as in the menu tree."""
    counts: "Counter[str]" = Counter()
    forms: DefaultDict[str, "Counter[str]"] = defaultdict(Counter)
    for values in docs_values:
        folded: Dict[str, str] = {}
        for value in sorted(values):
            group = value_group(value)
            if group:
                folded.setdefault(group, value)
        for key, value in folded.items():
            counts[key] += 1
            forms[key][key if key in ('present', 'absent') else value] += 1
    not_mentioned = n_results - len(docs_values)
    rows: List[Dict[str, Any]] = []
    for key, count in counts.items():
        shown = _most_common(forms[key])
        rows.append({
            'value': shown,
            'count': count,
            'others': 1 - count / n_results,
            'display_text': _display_text(path, shown),
        })
    rows.sort(key=lambda r: (-r['count'], r['value']))
    return {
        'path': list(path),
        'score': split_score(list(counts.values()) + [not_mentioned],
                             n_results),
        'not_mentioned': not_mentioned,
        'values': rows,
    }


def score_structured(docs: Sequence[Mapping[str, Any]],
                     top_n: Optional[int] = None) -> Dict[str, Any]:
    """Score the structured descriptions of the current results.

    Returns the F2 response shape: ``{"n_results": N, "parts": [...]}``,
    each part ``{"path", "described", "score", "attributes", "parts"}``
    and each attribute ``{"path", "score", "not_mentioned", "values"}``.
    A part's score is its best feature: its own "described" score, or
    the best of its attributes and nested parts.  ``top_n`` caps the
    number of attributes kept, best first, and a part left with no
    attribute beneath it is dropped.
    """
    n_results = len(docs)
    described: "Counter[Path]" = Counter()
    attr_values: DefaultDict[Path, List[Set[str]]] = defaultdict(list)
    for doc in docs:
        parts, attributes = structured_features(
            doc if isinstance(doc, Mapping) else {})
        prefixes: Set[Path] = set()
        for path in list(parts) + [_owner(a) for a in attributes]:
            prefixes.update(path[:i] for i in range(1, len(path) + 1))
        described.update(prefixes)
        for path, values in attributes.items():
            attr_values[path].append(values)

    scored = [_score_attribute(path, values, n_results)
              for path, values in attr_values.items()]
    scored.sort(key=lambda a: (-a['score'], a['path']))
    if top_n is not None:
        scored = scored[:max(top_n, 0)]

    by_owner: DefaultDict[Path, List[Dict[str, Any]]] = defaultdict(list)
    for attribute in scored:
        by_owner[_owner(tuple(attribute['path']))].append(attribute)

    keep: Set[Path] = set(described)
    if top_n is not None:
        keep = {p[:i] for p in by_owner for i in range(1, len(p) + 1)}

    def build(path: Path) -> Dict[str, Any]:
        children = [build(c) for c in keep
                    if len(c) == len(path) + 1 and c[:len(path)] == path]
        children.sort(key=lambda p: (-p['score'], p['path']))
        attributes = by_owner.get(path, [])
        score = max([feature_score(described[path], n_results)]
                    + [a['score'] for a in attributes]
                    + [c['score'] for c in children])
        return {'path': list(path), 'described': described[path],
                'score': score, 'attributes': attributes,
                'parts': children}

    top = [build(p) for p in keep if len(p) == 1]
    top.sort(key=lambda p: (-p['score'], p['path']))
    return {'n_results': n_results, 'parts': top}


def _fold(text: str) -> str:
    """Lowercase and strip accents, as TF-IDF's ``strip_accents``
    ``'unicode'`` does."""
    decomposed = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in decomposed if not unicodedata.combining(c))


def text_ngrams(text: str) -> Set[str]:
    """Return the word 1- and 2-grams of ``text``: lowercased, accents
    stripped, words of two or more word characters."""
    tokens = TOKEN_RE.findall(_fold(text))
    grams = set(tokens)
    grams.update(f'{a} {b}' for a, b in zip(tokens, tokens[1:]))
    return grams


def score_text(texts: Sequence[str],
               top_n: Optional[int] = None) -> Dict[str, Any]:
    """Score the n-grams of the current results' texts.

    Returns ``{"n_results": N, "features": [{"text", "count",
    "score"}, ...]}``, best first.  N-grams found in exactly the same
    results collapse to one row, the longest (most words, then most
    characters, then first alphabetically).
    """
    n_results = len(texts)
    found: DefaultDict[str, Set[int]] = defaultdict(set)
    for index, text in enumerate(texts):
        for gram in text_ngrams(text or ''):
            found[gram].add(index)
    groups: DefaultDict[FrozenSet[int], List[str]] = defaultdict(list)
    for gram, indices in found.items():
        groups[frozenset(indices)].append(gram)
    features: List[Dict[str, Any]] = []
    for results, grams in groups.items():
        longest = min(grams, key=lambda g: (-len(g.split()), -len(g), g))
        features.append({'text': longest, 'count': len(results),
                         'score': feature_score(len(results), n_results)})
    features.sort(key=lambda f: (-f['score'], -f['count'], f['text']))
    if top_n is not None:
        features = features[:max(top_n, 0)]
    return {'n_results': n_results, 'features': features}


__all__: List[str] = [
    'feature_score', 'score_structured', 'score_text', 'split_score',
    'structured_features', 'text_ngrams',
]
