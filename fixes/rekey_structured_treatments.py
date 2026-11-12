#!/usr/bin/env python3
"""Rename structured treatments to their prose treatment's ``_id``.

treatments_to_json used to name each structured treatment by a hash of
its treatment name and description-plus-diagnosis text, while prose
treatments are named by a hash of all eleven section fields.  The two
never match, so anything that looks a structured treatment up by the
prose ``_id`` it was given (the JSON Features tab) finds nothing.
treatments_to_json now names structured treatments by the prose
``_id``; this one-shot fix renames the existing ones.

For each translatable prose treatment, the old name is
``taxon_doc_id(treatment, build_llm_input_text(doc))``.  The document is
copied to the prose ``_id`` (prose treatments sharing identical text
each get their own copy); old names are deleted only after every copy
succeeded.  Rerunning finds nothing left to do.

Usage::

    fixes/rekey_structured_treatments.py --experiment production_v4_1 --dry-run
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))


def legacy_structured_id(prose_doc: Dict[str, Any]) -> str:
    """The name treatments_to_json gave this prose treatment's JSON."""
    raise NotImplementedError


def plan_rekey(
    prose_docs: Iterable[Dict[str, Any]],
    structured_ids: Set[str],
) -> Dict[str, Any]:
    """What renaming would do, without doing it.

    Returns ``{copies: [(old_id, new_id)], already: [new_id],
    missing: [prose_id], retire: {old_id}, untranslatable: int}``:

    - copies: prose treatments whose JSON exists under the old name only;
    - already: prose treatments already named by their own ``_id``;
    - missing: translatable prose treatments with no JSON under either;
    - retire: old names to delete once every copy is made;
    - untranslatable: prose treatments with no usable text (never sent).
    """
    raise NotImplementedError


def apply_rekey(
    db: Any,
    plan: Dict[str, Any],
    dry_run: bool = False,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Make the copies, then delete the retired names.

    A copy is the old document's content under the new ``_id`` (no
    ``_rev``).  If any copy fails nothing is deleted, so a rerun can
    finish the job.  Returns ``{copied, failed, deleted}``; under
    ``dry_run`` the counts are what would happen and nothing is written.
    """
    raise NotImplementedError


def target_dbs(experiment_name: Optional[str],
               config: Dict[str, Any]) -> Dict[str, str]:
    """``{prose, structured}`` database names for the experiment.

    Raises:
        ValueError: Without ``--experiment``; a rename must never fall
            back to a default database.
    """
    raise NotImplementedError


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
