#!/usr/bin/env python3
"""Bootstrap-pass Claude annotator: one feature, one Treatment per call.

Phase 1 deliverable 5 of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4.

Pipes from ``bin/select_for_annotation`` (item 2): reads treatment
IDs on stdin (or via ``--doc-id``), fetches each Treatment from
the experiment's ``treatments_prose`` DB, renders the synthetic
brat ``.txt`` (item 4's ``treatments_to_structured.brat_render.render``),
sends it to Claude with the feature schema (item 3) as a structural
prompt ingredient, parses the response into annotation docs
(item 5 part 1's ``parse_claude_response``), and writes them to
the experiment's candidate annotations DB.

The candidate DB name comes from the experiment doc's
``databases.features_candidate`` field (Phase 1 deliverable 4.5),
with a naming-convention fallback to
``skol_exp_<experiment>_features_candidate`` when 4.5 hasn't been
run yet.

Usage::

    # End-to-end pipeline:
    bin/select_for_annotation --experiment production_v4 --n 10 --seed 1 \\
        | bin/llm_annotate_features --experiment production_v4

    # Specific IDs:
    bin/llm_annotate_features --experiment production_v4 \\
        --doc-id taxon_abc,taxon_xyz

    # Token / cost estimate before committing budget:
    bin/llm_annotate_features --experiment production_v4 \\
        --doc-id taxon_abc --estimate

    # Different feature schema (when more land in schemas/):
    bin/llm_annotate_features --experiment production_v4 \\
        --schema lamellae --doc-id taxon_abc

Environment:
    ANTHROPIC_API_KEY — required for both --estimate (count_tokens)
        and live annotation.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402

from treatments_to_structured.brat_render import render  # noqa: E402
from treatments_to_structured.llm_annotate import (  # noqa: E402
    _SYSTEM_PROMPT,
    annotation_doc_id,
    build_user_prompt,
    parse_claude_response,
)


_DEFAULT_MODEL = 'claude-opus-4-7'
_DEFAULT_WORKERS = 5
_DEFAULT_MAX_TOKENS = 4096

_SCHEMAS_DIR = (
    Path(__file__).resolve().parent.parent
    / 'treatments_to_structured'
    / 'schemas'
)

# Pricing per 1M tokens (USD).  Mirrors bin/llm_relabel.py's _PRICING
# table; same values where the model overlaps, opus-4-7 / 4-8 added
# for Phase 1's default.
_PRICING: Dict[str, Dict[str, float]] = {
    'claude-haiku-4-5-20251001': {'input': 0.80, 'output': 4.00},
    'claude-sonnet-4-6': {'input': 3.00, 'output': 15.00},
    'claude-opus-4-6': {'input': 15.00, 'output': 75.00},
    'claude-opus-4-7': {'input': 15.00, 'output': 75.00},
    'claude-opus-4-8': {'input': 15.00, 'output': 75.00},
}


# ---------------------------------------------------------------------------
# Testable helpers
# ---------------------------------------------------------------------------


def load_schema(name: str) -> Dict[str, Any]:
    """Load a JSON Schema from ``treatments_to_structured/schemas/<name>.json``.

    Raises ``FileNotFoundError`` if the schema isn't present so the
    CLI can convert to a clean stderr message.
    """
    path = _SCHEMAS_DIR / f'{name}.json'
    with path.open('r') as f:
        return json.load(f)


def resolve_candidate_db_name(
    experiment_name: str,
    experiment_doc: Dict[str, Any],
    *,
    verbosity: int = 1,
    warn_stream: TextIO = sys.stderr,
) -> str:
    """Find the candidate DB name for an experiment.

    Prefers ``experiment.databases.features_candidate`` (Phase 1
    deliverable 4.5).  Falls back to the
    ``skol_exp_<name>_features_candidate`` naming convention with a
    one-line warning, so the script works on experiments that
    haven't been migrated yet.
    """
    dbs = experiment_doc.get('databases') or {}
    explicit = dbs.get('features_candidate')
    if explicit:
        return explicit
    fallback = f'skol_exp_{experiment_name}_features_candidate'
    if verbosity >= 1:
        print(
            f"NOTE: experiment.databases.features_candidate not set; "
            f"using naming-convention fallback {fallback!r}.  Run "
            f"`bin/manage_experiment update {experiment_name}` once "
            f"deliverable 4.5 lands to make this canonical.",
            file=warn_stream,
        )
    return fallback


def read_treatment_ids(
    doc_ids: Optional[List[str]],
    stdin_stream: TextIO,
    *,
    stdin_isatty: bool,
) -> List[str]:
    """Resolve treatment IDs from ``--doc-id`` (pre-parsed by
    common_parser) or stdin.

    Either source must produce at least one non-empty ID.  Stdin is
    consumed only when it isn't a TTY (otherwise the script would
    block waiting for typed input).

    Raises ``ValueError`` with an operator-actionable message if
    neither source yields IDs.
    """
    if doc_ids:
        # common_parser already split and stripped; filter empties
        # defensively in case anyone passed a list with blanks.
        ids = [s for s in doc_ids if s and s.strip()]
        if not ids:
            raise ValueError(
                "--doc-id contained no valid IDs after stripping"
            )
        return ids
    if not stdin_isatty:
        ids = [line.strip() for line in stdin_stream if line.strip()]
        if not ids:
            raise ValueError(
                "stdin was empty; no treatment IDs to process"
            )
        return ids
    raise ValueError(
        "no treatment IDs provided; pass --doc-id ID[,ID,...] or "
        "pipe IDs on stdin (e.g., from bin/select_for_annotation)"
    )


def filter_already_annotated(
    treatment_ids: Iterable[str],
    candidate_db: Any,
    feature_label: str,
) -> List[str]:
    """Return the subset of treatment IDs without existing
    annotations for ``feature_label`` in ``candidate_db``.

    Annotation docs are keyed ``<treatment_id>:<feature_label>:<offset>``
    so a single ``_all_docs`` range query per treatment ID tells us
    whether any annotation already exists.  Cheap on small candidate
    DBs; if this becomes hot at scale, swap to a view.
    """
    out: List[str] = []
    for tid in treatment_ids:
        prefix = f'{tid}:{feature_label}:'
        rows = candidate_db.view(
            '_all_docs', startkey=prefix, endkey=prefix + '￰',
            limit=1,
        ).rows
        if not rows:
            out.append(tid)
    return out


def estimate_tokens(
    client: Any,
    prompts: List[Tuple[str, str]],
    model: str,
    system_prompt: str = _SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Count input tokens via Anthropic ``count_tokens``; estimate
    output tokens and total cost.

    Output tokens are estimated at 1/4 of input — annotation output
    is a small JSON envelope (a few short text fields per span),
    typically much smaller than the prompt's schema-plus-treatment
    payload.  Calibrate after the first real run if the ratio
    proves systematically off.
    """
    total_input = 0
    for _treatment_id, user_prompt in prompts:
        result = client.messages.count_tokens(
            model=model,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        total_input += result.input_tokens
    est_output = total_input // 4
    pricing = _PRICING.get(
        model, {'input': 15.00, 'output': 75.00},
    )
    input_cost = total_input * pricing['input'] / 1_000_000
    output_cost = est_output * pricing['output'] / 1_000_000
    return {
        'doc_count': len(prompts),
        'total_input_tokens': total_input,
        'est_output_tokens': est_output,
        'est_total_tokens': total_input + est_output,
        'est_input_cost_usd': round(input_cost, 4),
        'est_output_cost_usd': round(output_cost, 4),
        'est_total_cost_usd': round(input_cost + output_cost, 4),
    }


def annotate_one_treatment(
    client: Any,
    treatment: Dict[str, Any],
    schema: Dict[str, Any],
    feature_label: str,
    model: str,
    system_prompt: str = _SYSTEM_PROMPT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> Union[List[Dict[str, Any]], Exception]:
    """Render, prompt, call Claude, parse — returns annotations or
    the exception that interrupted the pipeline.

    Returning the exception (rather than raising) keeps the parallel
    worker pool tidy: the outer loop logs per-treatment failures
    without unwinding healthy concurrent calls.
    """
    treatment_id = treatment['_id']
    doc_id = (treatment.get('ingest') or {}).get('_id') or ''
    try:
        synth_txt, span_map = render(treatment)
        if not synth_txt:
            return []
        user_prompt = build_user_prompt(synth_txt, schema, feature_label)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        response_text = response.content[0].text
        now = datetime.now(timezone.utc).isoformat()
        return parse_claude_response(
            response_text, span_map, feature_label, model,
            treatment_id, doc_id, now,
        )
    except Exception as exc:  # noqa: BLE001 — propagate per-worker
        return exc


def _print_estimate(stats: Dict[str, Any], model: str) -> None:
    """Pretty-print an estimate-mode summary to stdout."""
    pricing = _PRICING.get(
        model, {'input': 15.00, 'output': 75.00},
    )
    print(f"\nToken estimate for {stats['doc_count']} treatment(s):")
    print(
        f"  Input tokens:             "
        f"{stats['total_input_tokens']:>12,}"
    )
    print(
        f"  Output tokens (estimate): "
        f"{stats['est_output_tokens']:>12,}"
    )
    print(
        f"  Total tokens (estimate):  "
        f"{stats['est_total_tokens']:>12,}"
    )
    print(
        f"  Pricing: ${pricing['input']:.2f}/1M input, "
        f"${pricing['output']:.2f}/1M output"
    )
    print(
        f"  Estimated cost (USD):     "
        f"${stats['est_total_cost_usd']:>12,.4f}"
    )


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--schema', default='pileus', metavar='NAME',
        help=(
            'Feature schema name (looked up in '
            'treatments_to_structured/schemas/<NAME>.json).  '
            'Default: pileus.'
        ),
    )
    parser.add_argument(
        '--feature-label', default=None, metavar='LABEL',
        help=(
            "Annotation label (defaults to the schema's 'title', "
            "e.g., 'Pileus' for schemas/pileus.json)."
        ),
    )
    # --doc-id is provided by common_parser() (dest='doc_ids', already
    # parsed to List[str]).  Alternative input: pipe IDs on stdin.
    parser.add_argument(
        '--llm-model', dest='model', default=_DEFAULT_MODEL,
        metavar='MODEL',
        help=f'Claude model ID (default: {_DEFAULT_MODEL}).',
    )
    parser.add_argument(
        '--workers', type=int, default=_DEFAULT_WORKERS, metavar='N',
        help=f'Parallel API workers (default: {_DEFAULT_WORKERS}).',
    )
    parser.add_argument(
        '--estimate', action='store_true',
        help=(
            'Count input tokens and estimate cost without generating '
            'output.  Use this before a full run to check budget.'
        ),
    )
    parser.add_argument(
        '--log-file', default=None, metavar='FILE',
        help=(
            'JSONL per-treatment log (default: '
            'llm_annotate_<timestamp>.jsonl).'
        ),
    )
    parser.add_argument(
        '--max-tokens', type=int, default=_DEFAULT_MAX_TOKENS,
        metavar='N',
        help=(
            f'Max output tokens per Claude call (default: '
            f'{_DEFAULT_MAX_TOKENS}).'
        ),
    )
    args = parser.parse_args()

    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    skip_existing = bool(config.get('skip_existing', False))
    experiment = config.get('experiment')
    if not experiment:
        print("error: --experiment is required", file=sys.stderr)
        return 2

    # Schema
    try:
        schema = load_schema(args.schema)
    except FileNotFoundError:
        print(
            f"error: schema {args.schema!r} not found in "
            f"{_SCHEMAS_DIR}",
            file=sys.stderr,
        )
        return 2
    feature_label = (
        args.feature_label
        or schema.get('title')
        or args.schema.title()
    )

    # Treatment IDs (--doc-id resolves to args.doc_ids via common_parser)
    try:
        treatment_ids = read_treatment_ids(
            args.doc_ids, sys.stdin, stdin_isatty=sys.stdin.isatty(),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Anthropic client + key check
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print(
            "error: ANTHROPIC_API_KEY environment variable not set",
            file=sys.stderr,
        )
        return 2
    try:
        import anthropic
    except ImportError:
        print(
            "error: anthropic package not installed "
            "(pip install anthropic)",
            file=sys.stderr,
        )
        return 2
    client = anthropic.Anthropic(api_key=api_key)

    # CouchDB
    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )

    treatments_db_name = (
        config.get('treatments_prose_db_name')
        or config.get('treatments_db_name')
    )
    if not treatments_db_name or treatments_db_name not in server:
        print(
            f"error: treatments_prose DB {treatments_db_name!r} "
            f"not found on the server",
            file=sys.stderr,
        )
        return 2
    treatments_db = server[treatments_db_name]

    # Resolve experiment doc → candidate DB
    try:
        exp_doc = server['skol_experiments'][experiment]
    except Exception:
        print(
            f"error: experiment {experiment!r} not found in "
            f"skol_experiments",
            file=sys.stderr,
        )
        return 2
    candidate_db_name = resolve_candidate_db_name(
        experiment, exp_doc, verbosity=verbosity,
    )
    if candidate_db_name not in server:
        if dry_run:
            if verbosity >= 1:
                print(
                    f"[dry-run] would create candidate DB "
                    f"{candidate_db_name!r}",
                    file=sys.stderr,
                )
            candidate_db = None
        else:
            if verbosity >= 1:
                print(
                    f"Creating candidate DB {candidate_db_name!r}",
                    file=sys.stderr,
                )
            candidate_db = server.create(candidate_db_name)
    else:
        candidate_db = server[candidate_db_name]

    # Skip-existing filter
    if skip_existing and candidate_db is not None:
        before = len(treatment_ids)
        treatment_ids = filter_already_annotated(
            treatment_ids, candidate_db, feature_label,
        )
        if verbosity >= 1:
            skipped = before - len(treatment_ids)
            if skipped > 0:
                print(
                    f"--skip-existing: dropped {skipped} treatments "
                    f"already annotated for {feature_label}",
                    file=sys.stderr,
                )

    # Fetch the actual Treatment docs
    if verbosity >= 1:
        print(
            f"Fetching {len(treatment_ids)} treatments from "
            f"{treatments_db_name}...",
            file=sys.stderr,
        )
    treatments: List[Dict[str, Any]] = []
    for tid in treatment_ids:
        try:
            treatments.append(dict(treatments_db[tid]))
        except Exception:
            if verbosity >= 1:
                print(
                    f"  skipping {tid}: not found in "
                    f"{treatments_db_name}",
                    file=sys.stderr,
                )

    if not treatments:
        print("No treatments to process.", file=sys.stderr)
        return 0

    # --estimate mode: build prompts, count tokens, exit
    if args.estimate:
        if verbosity >= 1:
            print(
                f"Building prompts for {len(treatments)} treatments "
                f"to estimate tokens...",
                file=sys.stderr,
            )
        prompts: List[Tuple[str, str]] = []
        for t in treatments:
            synth_txt, _ = render(t)
            if not synth_txt:
                continue
            prompts.append((
                t['_id'],
                build_user_prompt(synth_txt, schema, feature_label),
            ))
        if not prompts:
            print(
                "No treatments with content to estimate.",
                file=sys.stderr,
            )
            return 0
        stats = estimate_tokens(client, prompts, args.model)
        _print_estimate(stats, args.model)
        return 0

    # Live annotation loop
    log_path = (
        args.log_file
        or f"llm_annotate_{int(time.time())}.jsonl"
    )
    success_count = 0
    error_count = 0
    empty_count = 0

    if verbosity >= 1:
        print(
            f"Annotating {len(treatments)} treatments "
            f"with {args.workers} workers; log: {log_path}",
            file=sys.stderr,
        )

    with open(log_path, 'a') as log_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    annotate_one_treatment, client, t, schema,
                    feature_label, args.model, _SYSTEM_PROMPT,
                    args.max_tokens,
                ): t['_id']
                for t in treatments
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                result = fut.result()
                if isinstance(result, Exception):
                    error_count += 1
                    if verbosity >= 1:
                        print(
                            f"  ERROR {tid}: {result}",
                            file=sys.stderr,
                        )
                    log_f.write(json.dumps({
                        'treatment_id': tid,
                        'status': 'error',
                        'error': str(result),
                    }) + '\n')
                    continue
                anns = result
                if not anns:
                    empty_count += 1
                    if verbosity >= 2:
                        print(
                            f"  {tid}: no annotations",
                            file=sys.stderr,
                        )
                    log_f.write(json.dumps({
                        'treatment_id': tid,
                        'status': 'empty',
                    }) + '\n')
                    continue
                if not dry_run and candidate_db is not None:
                    for ann in anns:
                        ann['_id'] = annotation_doc_id(
                            tid, feature_label, ann['start'],
                        )
                        if ann['_id'] in candidate_db:
                            ann['_rev'] = (
                                candidate_db[ann['_id']]['_rev']
                            )
                        try:
                            candidate_db.save(ann)
                        except Exception as exc:
                            error_count += 1
                            if verbosity >= 1:
                                print(
                                    f"  ERROR saving {ann['_id']}: "
                                    f"{exc}",
                                    file=sys.stderr,
                                )
                success_count += 1
                if verbosity >= 1:
                    print(
                        f"  {tid}: {len(anns)} annotation(s)",
                        file=sys.stderr,
                    )
                log_f.write(json.dumps({
                    'treatment_id': tid,
                    'status': 'success',
                    'n_annotations': len(anns),
                }) + '\n')

    print(
        f"\nDone: {success_count} with annotations, "
        f"{empty_count} empty, {error_count} errors"
    )
    print(f"Log: {log_path}")
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
