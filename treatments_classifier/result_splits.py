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

This module is a skeleton: every function raises
``NotImplementedError`` until the tests in ``result_splits_test.py``
are approved and implemented.
"""

from typing import (Any, Dict, FrozenSet, List, Mapping, Optional,
                    Sequence, Set, Tuple)

Path = Tuple[str, ...]


def split_score(counts: Sequence[int], n_results: int) -> float:
    """Return the expected share of ``n_results`` that differ from an
    answer, given each answer's count of results.

    Counts may sum to more than ``n_results`` (a result can give
    several answers).  Zero results, or all-zero counts, score 0.
    """
    raise NotImplementedError


def feature_score(count: int, n_results: int) -> float:
    """Score a single yes/no feature present in ``count`` results:
    ``2k(N - k) / N**2``."""
    raise NotImplementedError


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
    """
    raise NotImplementedError


def score_structured(docs: Sequence[Mapping[str, Any]],
                     top_n: Optional[int] = None) -> Dict[str, Any]:
    """Score the structured descriptions of the current results.

    Returns the F2 response shape: ``{"n_results": N, "parts": [...]}``,
    each part ``{"path", "described", "score", "attributes", "parts"}``
    and each attribute ``{"path", "score", "not_mentioned", "values"}``.
    ``top_n`` caps the number of attributes kept, best first.
    """
    raise NotImplementedError


def text_ngrams(text: str) -> Set[str]:
    """Return the word 1- and 2-grams of ``text``: lowercased, accents
    stripped, words of two or more word characters."""
    raise NotImplementedError


def score_text(texts: Sequence[str],
               top_n: Optional[int] = None) -> Dict[str, Any]:
    """Score the n-grams of the current results' texts.

    Returns ``{"n_results": N, "features": [{"text", "count",
    "score"}, ...]}``, best first.  N-grams found in exactly the same
    results collapse to one row, the longest.
    """
    raise NotImplementedError


__all__: List[str] = [
    'feature_score', 'score_structured', 'score_text', 'split_score',
    'structured_features', 'text_ngrams',
]
