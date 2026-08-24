#!/usr/bin/env python3
"""Bootstrap-pass Claude annotator: all features for one Treatment per call.

Phase 1 deliverable 5 of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4.

Pipes from ``bin/select_for_annotation`` (item 2): reads treatment
IDs on stdin (or via ``--doc-id``), fetches each Treatment from
the experiment's ``treatments_prose`` DB, renders the synthetic
brat ``.txt`` (item 4's ``treatments_to_structured.brat_render.render``),
sends it to Claude with the feature SEED (item 3, e.g.
``seeds/fungi.json``) as an open-ended example vocabulary, parses
the response into annotation docs (item 5 part 1's
``parse_claude_response``), and writes them to the experiment's
candidate annotations DB.

One Claude call per Treatment returns spans for ALL anatomical
features it identifies, each tagged with its own
``feature_label``.  Labels not in the seed are accepted — Claude
is instructed to invent canonical anatomical names for them.
Cross-kingdom generalization is a seed swap: ``--feature-set
plants`` would point at a future ``seeds/plants.json``.

The candidate DB name comes from the experiment doc's
``databases.features_candidate`` field (Phase 1 deliverable 4.5),
with a naming-convention fallback to
``skol_exp_<experiment>_02_50_features_candidate`` when 4.5
hasn't been run on the experiment.

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

    # Different feature seed (when more land in seeds/):
    bin/llm_annotate_features --experiment plant_treatments_v1 \\
        --feature-set plants --doc-id specimen_abc

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
from typing import Any, Dict, Iterable, Iterator, List, Optional, TextIO, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402
from llm_pricing import PRICING  # noqa: E402

from treatments_to_structured.brat_render import render  # noqa: E402
from treatments_to_structured.complexity import (  # noqa: E402
    complexity_score,
)
from treatments_to_structured.llm_annotate import (  # noqa: E402
    _SYSTEM_PROMPT,
    ClaudeResponseError,
    annotation_doc_id,
    build_user_prompt,
    parse_claude_response,
)
from treatments_to_structured.status import (  # noqa: E402
    STATUS_ERROR,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
    AnnotationResult,
    classify_result,
    make_status_doc,
    status_doc_id,
)


_DEFAULT_MODEL = 'claude-opus-4-7'
_DEFAULT_WORKERS = 5
# 4096 was too tight on large treatments (Amanita magniverrucata
# triggered output truncation 2026-06-29: Claude began emitting
# a 10KB description verbatim as the first span's `text` and ran
# out of tokens partway through, producing un-fenced JSON that
# failed the parser).  16384 leaves headroom for treatments with
# 30+ annotations + long greedy text spans.  Operator override:
# pass --max-tokens.
_DEFAULT_MAX_TOKENS = 16384

_SEEDS_DIR = (
    Path(__file__).resolve().parent.parent
    / 'treatments_to_structured'
    / 'seeds'
)


# ---------------------------------------------------------------------------
# Testable helpers
# ---------------------------------------------------------------------------


def load_seed(name: str) -> Dict[str, Any]:
    """Load a feature-seed file from
    ``treatments_to_structured/seeds/<name>.json``.

    A seed file is the bootstrap annotator's open-ended example
    list of anatomical feature labels (see ``seeds/fungi.json`` for
    the worked example).  Not exhaustive — Claude is expected to
    invent canonical labels for features the seed doesn't list.

    Raises ``FileNotFoundError`` if the seed isn't present so the
    CLI can convert to a clean stderr message.
    """
    path = _SEEDS_DIR / f'{name}.json'
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
    # 02_50 slots between the 02_00 treatments_prose extraction and
    # the 03_00 treatments_structured SLM output, per the
    # sort-in-pipeline-order convention from
    # docs/skol-db-naming-cleanup.md (also memory:
    # project_db_naming_cleanup.md).
    fallback = (
        f'skol_exp_{experiment_name}_02_50_features_candidate'
    )
    if verbosity >= 1:
        print(
            f"NOTE: experiment.databases.features_candidate not set; "
            f"using naming-convention fallback {fallback!r}.  Run "
            f"`bin/manage_experiment update {experiment_name}` once "
            f"deliverable 4.5 lands to make this canonical.",
            file=warn_stream,
        )
    return fallback


STDIN_SENTINEL = '-'


def iter_treatment_ids(stream: TextIO) -> Iterator[str]:
    """Yield treatment ids from ``stream``, one per ``readline``.

    The streaming primitive behind ``--doc-id -``.  Blank lines are
    skipped; ``''`` (EOF) terminates.

    **Uses ``readline``, deliberately, not iteration.**  ``for line in
    sys.stdin`` block-buffers when stdin is a pipe, so an id pasted
    into a live review window would sit unprocessed until the buffer
    filled — correct-looking against a TTY and useless in practice.
    """
    while True:
        line = stream.readline()
        if not line:          # '' is EOF; '\n' is a blank line
            return
        text = line.strip()
        if text:
            yield text


def resolve_id_filter(
    doc_ids: Optional[List[str]],
    stdin_stream: TextIO,
    *,
    stdin_isatty: bool,
) -> Optional[List[str]]:
    """Resolve an *optional* id filter, honouring the ``-`` sentinel.

    For tools where "no filter" is a legitimate default —
    ``brat_export`` and ``brat_ingest`` both process a whole
    ``--ann-dir`` when unrestricted — as opposed to
    :func:`read_treatment_ids`, which requires ids and raises without
    them.

    * ``None`` / empty  → ``None`` (no filter).  **Stdin is not read.**
    * ``['-']``         → the ids on stdin, as a batch.
    * anything else     → returned unchanged.

    The stdin-not-read case is the one that matters: cron supplies a
    non-TTY ``/dev/null`` stdin, so a tool that read stdin whenever it
    was not a TTY would consume nothing and process nothing, silently
    breaking every scheduled invocation.
    """
    if not doc_ids:
        return None
    if list(doc_ids) == [STDIN_SENTINEL]:
        return read_treatment_ids(
            None, stdin_stream, stdin_isatty=stdin_isatty,
        )
    return list(doc_ids)


def read_treatment_ids(
    doc_ids: Optional[List[str]],
    stdin_stream: TextIO,
    *,
    stdin_isatty: bool,
) -> List[str]:
    """Resolve treatment IDs from ``--doc-id`` (pre-parsed by
    ``get_env_config``) or stdin.

    Either source must produce at least one non-empty ID.  Stdin is
    consumed only when it isn't a TTY (otherwise the script would
    block waiting for typed input).

    Raises ``ValueError`` with an operator-actionable message if
    neither source yields IDs.  Raises ``TypeError`` if ``doc_ids``
    is a string (the un-parsed form from ``args.doc_ids``) — silent
    character-iteration was the symptom of the first run's bug;
    failing loudly catches a recurrence immediately.
    """
    if doc_ids is not None and not isinstance(doc_ids, list):
        raise TypeError(
            f"doc_ids must be a list (or None), not "
            f"{type(doc_ids).__name__}.  Pass config['doc_ids'] from "
            f"get_env_config, NOT args.doc_ids (which is the raw "
            f"comma-separated string from argparse)."
        )
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
        # Drained through the streaming primitive so the two forms
        # cannot drift on blank-line / EOF handling.
        ids = list(iter_treatment_ids(stdin_stream))
        if not ids:
            raise ValueError(
                "stdin was empty; no treatment IDs to process"
            )
        return ids
    raise ValueError(
        "no treatment IDs provided; pass --doc-id ID[,ID,...] or "
        "pipe IDs on stdin (e.g., from bin/select_for_annotation)"
    )


def resolve_status_db_name(
    experiment_name: str,
    experiment_doc: Dict[str, Any],
    *,
    verbosity: int = 1,
    warn_stream: TextIO = sys.stderr,
) -> str:
    """Find the status DB name for an experiment.

    Mirrors ``resolve_candidate_db_name``: prefers
    ``experiment.databases.features_status``, falls back to the
    ``skol_exp_<name>_02_50_features_status`` naming convention
    with a one-line warning for experiments that haven't been
    migrated yet.

    Sibling DB to ``features_candidate`` — Design Y from the
    per-span-isolation discussion.  See
    ``treatments_to_structured/status.py`` for the doc schema.
    """
    dbs = experiment_doc.get('databases') or {}
    explicit = dbs.get('features_status')
    if explicit:
        return explicit
    fallback = (
        f'skol_exp_{experiment_name}_02_50_features_status'
    )
    if verbosity >= 1:
        print(
            f"NOTE: experiment.databases.features_status not set; "
            f"using naming-convention fallback {fallback!r}.  Run "
            f"`bin/manage_experiment update {experiment_name}` to "
            f"make this canonical.",
            file=warn_stream,
        )
    return fallback


def filter_already_annotated(
    treatment_ids: Iterable[str],
    status_db: Any,
    *,
    mode: str = 'default',
) -> List[str]:
    """Return the subset of treatment IDs that should be processed
    based on their status doc state.

    Three modes (mapped from CLI flags):

      * ``'default'`` — skip treatments with status ``success``.
        Retry ``error`` and ``partial`` (the offline-recovery
        script might have updated them, or a re-run with fresh
        prompt might succeed).
      * ``'skip_existing'`` — skip treatments with ANY status
        doc, regardless of state.  Operator override for "don't
        spend API budget on retries right now."
      * ``'force'`` — re-process everything; ignore status docs
        entirely.  Used after a prompt change.

    Returns the surviving IDs in the original order.  Status DB
    lookups are individual ``__getitem__`` calls; cheap on a
    treatment-id-keyed doc (no view scan).
    """
    if mode == 'force':
        return list(treatment_ids)
    out: List[str] = []
    for tid in treatment_ids:
        doc_id = status_doc_id(tid)
        try:
            status_doc = status_db[doc_id]
        except Exception:  # noqa: BLE001 — couchdb.ResourceNotFound
            # No status doc → never attempted → process.
            out.append(tid)
            continue
        existing_status = status_doc.get('status')
        if mode == 'skip_existing':
            # Any status doc → skip.
            continue
        # mode == 'default': retry error/partial, skip success.
        if existing_status == STATUS_SUCCESS:
            continue
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

    Output tokens are estimated at 1/2 of input.  Calibrated 2026-06-29
    from 6 live treatments under the multi-feature bootstrap prompt:
    measured output/input ratio was 47.12%.  The original estimate
    (1/4) was a guess from before the multi-feature pivot —
    annotations now greedily echo whole feature blocks as
    ``source_text``, so the output is much closer to the input size
    than the old per-feature variant.  Re-calibrate from the
    notebook's metrics cell when the corpus grows.
    """
    total_input = 0
    for _treatment_id, user_prompt in prompts:
        result = client.messages.count_tokens(
            model=model,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        total_input += result.input_tokens
    est_output = total_input // 2
    pricing = PRICING.for_model(model)
    input_cost = total_input * pricing.input / 1_000_000
    output_cost = est_output * pricing.output / 1_000_000
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
    seed: Dict[str, Any],
    model: str,
    system_prompt: str = _SYSTEM_PROMPT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> AnnotationResult:
    """Render, prompt, call Claude, parse — return one
    ``AnnotationResult`` per Treatment regardless of outcome.

    Returns an ``AnnotationResult`` with one of three statuses:

      * ``success`` — every span Claude returned was recovered.
        Includes the legitimate "Claude returned no spans" case
        (annotations is empty list).
      * ``partial`` — Claude returned at least one span we
        couldn't recover (e.g., unicode normalization too
        aggressive, span crossed a field boundary).
        ``dropped_spans`` lists them for the offline-recovery
        fixes/ script.  ``annotations`` may also be non-empty
        with the spans that DID recover.
      * ``error`` — catastrophic failure before any spans could
        be parsed (Anthropic API error, invalid JSON, envelope
        violations, network failure).  ``error_message`` carries
        the diagnostic.

    Never raises — the parallel worker pool stays healthy.  Per-
    treatment failures show up as ``status='error'`` results in
    the main loop, alongside the successful results.
    """
    treatment_id = treatment['_id']
    doc_id = (treatment.get('ingest') or {}).get('_id') or ''
    # Instrumentation — collected as we go and stamped into the
    # status doc's `metrics` sub-dict.  Captures everything needed
    # to plot Heaps' Law (label growth) and the cost / perf
    # regressions in the analysis notebook.
    wall_clock_start = time.monotonic()
    metrics: Dict[str, Any] = {
        'complexity_score': complexity_score(treatment),
        'synth_doc_chars': 0,
        'api_latency_seconds': None,
        'input_tokens': None,
        'output_tokens': None,
        'wall_clock_seconds': None,
    }
    try:
        synth_txt, span_map = render(treatment)
        metrics['synth_doc_chars'] = len(synth_txt)
        if not synth_txt:
            # Empty synth doc — Treatment had no description /
            # diagnosis.  Legitimate success: there were no
            # features to find, so we "found" none.  No API call
            # was made, so token / latency stay None.
            metrics['wall_clock_seconds'] = (
                time.monotonic() - wall_clock_start
            )
            return AnnotationResult(
                treatment_id=treatment_id,
                status=STATUS_SUCCESS,
                metrics=metrics,
            )
        user_prompt = build_user_prompt(synth_txt, seed)
        api_start = time.monotonic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        metrics['api_latency_seconds'] = (
            time.monotonic() - api_start
        )
        # response.usage is the Anthropic SDK's measured token
        # counts.  Replaces our 1/4-of-input output-token
        # estimate with the real value, so cost analysis is
        # exact rather than guessed.  Guarded with getattr so
        # tests that mock the response don't have to add usage.
        usage = getattr(response, 'usage', None)
        if usage is not None:
            metrics['input_tokens'] = getattr(
                usage, 'input_tokens', None,
            )
            metrics['output_tokens'] = getattr(
                usage, 'output_tokens', None,
            )
        # Capture stop_reason for diagnostics — when Claude returns
        # empty content (e.g., refusal on corrupt-input treatments
        # with garbled text), stop_reason names the cause.
        stop_reason = getattr(response, 'stop_reason', None)
        if stop_reason is not None:
            metrics['stop_reason'] = stop_reason
        # Defensive: Claude may return an empty content list when
        # the input triggers a refusal (e.g., the 2026-06-29
        # Colletotrichum treatment whose diagnosis was hundreds of
        # U+FFFD replacement chars from corrupt OCR — Claude
        # responded with 1 output token and content=[], which used
        # to raise IndexError here).  Surface as a clean error with
        # actionable diagnostic instead.
        if not response.content:
            metrics['wall_clock_seconds'] = (
                time.monotonic() - wall_clock_start
            )
            return AnnotationResult(
                treatment_id=treatment_id,
                status=STATUS_ERROR,
                error_message=(
                    f'Claude returned empty content '
                    f'(stop_reason={stop_reason!r}, '
                    f'output_tokens={metrics.get("output_tokens")}). '
                    f'Often signals an input the model refused — '
                    f'check the treatment for corrupt text '
                    f'(replacement chars, OCR gibberish).'
                ),
                metrics=metrics,
            )
        response_text = response.content[0].text
        now = datetime.now(timezone.utc).isoformat()
        annotations, dropped_spans = parse_claude_response(
            response_text, span_map, model,
            treatment_id, doc_id, now,
        )
        metrics['wall_clock_seconds'] = (
            time.monotonic() - wall_clock_start
        )
        return AnnotationResult(
            treatment_id=treatment_id,
            status=classify_result(annotations, dropped_spans, None),
            annotations=annotations,
            dropped_spans=dropped_spans,
            metrics=metrics,
        )
    except ClaudeResponseError as exc:
        # Envelope-level failure (invalid JSON, missing 'spans'
        # key, span missing 'text'/'feature_label').  Recovery
        # requires a re-prompt; classify as error so retry logic
        # picks it up.
        metrics['wall_clock_seconds'] = (
            time.monotonic() - wall_clock_start
        )
        return AnnotationResult(
            treatment_id=treatment_id,
            status=STATUS_ERROR,
            error_message=f'{type(exc).__name__}: {exc}',
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 — propagate per-worker
        # Anthropic API error, network failure, anything else.
        # Same classification — error doc with diagnostic, parallel
        # workers continue.  Metrics may be partial (e.g., API call
        # never returned, so input_tokens stays None) — that's a
        # useful signal too.
        metrics['wall_clock_seconds'] = (
            time.monotonic() - wall_clock_start
        )
        return AnnotationResult(
            treatment_id=treatment_id,
            status=STATUS_ERROR,
            error_message=f'{type(exc).__name__}: {exc}',
            metrics=metrics,
        )


def _print_estimate(stats: Dict[str, Any], model: str) -> None:
    """Pretty-print an estimate-mode summary to stdout."""
    pricing = PRICING.for_model(model)
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
        f"  Pricing: ${pricing.input:.2f}/1M input, "
        f"${pricing.output:.2f}/1M output"
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
        '--feature-set', default='fungi', metavar='NAME',
        help=(
            'Seed-vocabulary file (looked up in '
            'treatments_to_structured/seeds/<NAME>.json).  Defines '
            "the open-ended example labels for the bootstrap pass.  "
            "Default: fungi.  Swap to seeds/plants.json (or wherever "
            "future seed files land) for non-fungal corpora."
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
    # --skip-existing and --force are both provided by common_parser().
    # Their semantics in this script:
    #   default (neither flag): skip status=success, retry error/partial
    #   --skip-existing:        skip ANY treatment with a status doc
    #   --force:                ignore status DB entirely; re-process all
    # See filter_already_annotated() for the rules.
    args = parser.parse_args()

    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    skip_existing = bool(config.get('skip_existing', False))
    experiment = config.get('experiment_name')
    if not experiment:
        print("error: --experiment is required", file=sys.stderr)
        return 2

    # Seed (open-ended feature-label vocabulary)
    try:
        seed = load_seed(args.feature_set)
    except FileNotFoundError:
        print(
            f"error: seed {args.feature_set!r} not found in "
            f"{_SEEDS_DIR}",
            file=sys.stderr,
        )
        return 2

    # Treatment IDs (--doc-id is parsed from string to list by
    # get_env_config; use config['doc_ids'], NOT args.doc_ids which
    # is still the raw comma-separated string).
    try:
        treatment_ids = read_treatment_ids(
            config.get('doc_ids'), sys.stdin,
            stdin_isatty=sys.stdin.isatty(),
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
    status_db_name = resolve_status_db_name(
        experiment, exp_doc, verbosity=verbosity,
    )

    def _get_or_create(name: str, kind: str) -> Any:
        """Create the DB on demand; honor --dry-run by returning None."""
        if name in server:
            return server[name]
        if dry_run:
            if verbosity >= 1:
                print(
                    f"[dry-run] would create {kind} DB {name!r}",
                    file=sys.stderr,
                )
            return None
        if verbosity >= 1:
            print(
                f"Creating {kind} DB {name!r}", file=sys.stderr,
            )
        return server.create(name)

    candidate_db = _get_or_create(candidate_db_name, 'candidate')
    status_db = _get_or_create(status_db_name, 'status')

    # Status-aware skip / retry filter.  Default behavior: retry
    # error and partial treatments, skip success.  --skip-existing
    # widens skip to any status doc.  --force ignores status
    # entirely.  See filter_already_annotated for the full rules.
    if status_db is not None:
        # Read --force from config so env var / experiment doc
        # overrides work the same way --skip-existing already does.
        force_mode = bool(config.get('force', False))
        if force_mode:
            mode = 'force'
        elif skip_existing:
            mode = 'skip_existing'
        else:
            mode = 'default'
        before = len(treatment_ids)
        treatment_ids = filter_already_annotated(
            treatment_ids, status_db, mode=mode,
        )
        if verbosity >= 1:
            skipped = before - len(treatment_ids)
            if skipped > 0:
                print(
                    f"status filter (mode={mode}): dropped "
                    f"{skipped} treatments",
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
                build_user_prompt(synth_txt, seed),
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
    counts: Dict[str, int] = {
        STATUS_SUCCESS: 0, STATUS_PARTIAL: 0, STATUS_ERROR: 0,
    }
    total_annotations_stored = 0
    total_spans_dropped = 0

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
                    annotate_one_treatment, client, t, seed,
                    args.model, _SYSTEM_PROMPT, args.max_tokens,
                ): t['_id']
                for t in treatments
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                result: AnnotationResult = fut.result()
                counts[result.status] = counts.get(
                    result.status, 0,
                ) + 1
                total_spans_dropped += len(result.dropped_spans)

                # Persist surviving annotations.  Even on
                # status='partial' we save what we got; the
                # status doc carries the dropped_spans queue
                # for offline recovery.
                if not dry_run and candidate_db is not None:
                    for ann in result.annotations:
                        ann['_id'] = annotation_doc_id(
                            tid, ann['feature_label'], ann['start'],
                        )
                        if ann['_id'] in candidate_db:
                            ann['_rev'] = (
                                candidate_db[ann['_id']]['_rev']
                            )
                        try:
                            candidate_db.save(ann)
                            total_annotations_stored += 1
                        except Exception as exc:
                            if verbosity >= 1:
                                print(
                                    f"  ERROR saving {ann['_id']}: "
                                    f"{exc}",
                                    file=sys.stderr,
                                )

                # Persist the per-treatment status doc (Design Y).
                # Always written: success, partial, AND error.
                # Operators can query the status DB to see what
                # was attempted, what failed, and what's eligible
                # for offline recovery.
                if not dry_run and status_db is not None:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    prior_attempts = 0
                    prior_rev = None
                    try:
                        prior_doc = status_db[status_doc_id(tid)]
                        prior_attempts = int(
                            prior_doc.get('attempt_count', 0)
                        )
                        prior_rev = prior_doc.get('_rev')
                    except Exception:  # noqa: BLE001
                        pass
                    status_doc = make_status_doc(
                        result, args.model, now_iso,
                        attempt_count=prior_attempts + 1,
                    )
                    if prior_rev is not None:
                        status_doc['_rev'] = prior_rev
                    try:
                        status_db.save(status_doc)
                    except Exception as exc:
                        if verbosity >= 1:
                            print(
                                f"  ERROR saving status for {tid}: "
                                f"{exc}",
                                file=sys.stderr,
                            )

                # Per-treatment stderr line — same shape as before
                # plus an explicit (drop) note when partial.
                if verbosity >= 1:
                    if result.status == STATUS_ERROR:
                        print(
                            f"  ERROR {tid}: {result.error_message}",
                            file=sys.stderr,
                        )
                    elif result.status == STATUS_PARTIAL:
                        print(
                            f"  PARTIAL {tid}: "
                            f"{len(result.annotations)} stored, "
                            f"{len(result.dropped_spans)} dropped",
                            file=sys.stderr,
                        )
                    elif result.annotations:
                        print(
                            f"  {tid}: "
                            f"{len(result.annotations)} annotation(s)",
                            file=sys.stderr,
                        )
                    elif verbosity >= 2:
                        print(
                            f"  {tid}: no annotations",
                            file=sys.stderr,
                        )

                log_f.write(json.dumps({
                    'treatment_id': tid,
                    'status': result.status,
                    'n_annotations': len(result.annotations),
                    'n_dropped': len(result.dropped_spans),
                    'error': result.error_message,
                }) + '\n')

    # End-of-run summary (always printed, even at verbosity=0).
    # The dropped-span line is the operator's signal that an
    # offline-recovery fixes/ script run is worthwhile.
    print(
        f"\nDone: {counts[STATUS_SUCCESS]} success, "
        f"{counts[STATUS_PARTIAL]} partial, "
        f"{counts[STATUS_ERROR]} error"
    )
    print(
        f"      {total_annotations_stored} annotations stored, "
        f"{total_spans_dropped} spans dropped"
    )
    if total_spans_dropped > 0:
        print(
            f"      (run an offline-recovery script against "
            f"{status_db_name} to retry dropped spans)"
        )
    print(f"Log: {log_path}")
    return 0 if counts[STATUS_ERROR] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
