# Argparse strict mode for `bin/` scripts

**Status: implemented (2026-06-23).** All `bin/` CLIs now reject unknown
flags. This document records the design and the rules for keeping it
that way.

## Problem

`bin/` scripts used `parser.parse_known_args()`, which silently swallows
unknown flags instead of raising `SystemExit`. This bit us at least
twice:

- `bin/embed_treatments.py --doc-id <id> --force` silently dropped
  `--doc-id <id>` and ran a ~50 min full-database encoding (the expected
  single-doc mode didn't exist; the script accepted the typo).
- A `--rate-limit-max_ms` typo (underscore vs. dash) and a no-op
  `--skip-existing` were both swallowed without complaint.

The common shape: someone (operator or AI writing a runbook) believes a
flag does something, the script accepts it without protest, and the
behavior is the *default* — usually a long, expensive operation.

### Why a naïve `parse_args()` swap was not enough

The obvious fix — replace `parse_known_args()` with `parse_args()` —
would have broken **every** script, because there were *two* parsers:

1. each script's own `ArgumentParser`, and
2. a second parser inside `env_config.get_env_config()`, which ran its
   own `ArgumentParser(...).parse_known_args()` over `sys.argv` to read
   the shared flags (`--couchdb-url`, `--dry-run`, `--limit`, …).

Each parser used `parse_known_args()` and ignored what the other owned.
A flag unknown to *both* fell through both — the actual bug. Making
either one strict in isolation makes it reject the *other's* flags.

## Design: one parser, shared via `parents=`

`env_config` exposes `common_parser()` — an
`ArgumentParser(add_help=False)` holding every shared flag. Each script
makes it a parent and parses strictly **once**:

```python
from env_config import common_parser, get_env_config

parser = argparse.ArgumentParser(
    parents=[common_parser()],
    description=...,
)
parser.add_argument('--script-specific-flag', ...)
args = parser.parse_args()                 # strict: one parser knows everything
config = get_env_config(cli_args=args)     # reuse the parsed namespace; no re-parse
```

`get_env_config(cli_args=args)` consumes the already-parsed namespace
instead of re-reading `sys.argv`. One parser owns every flag, so an
unknown flag is rejected with `error: unrecognized arguments` and exit
code 2.

Three properties fall out for free:

- **Typo rejection** — the whole point.
- **Identical shared parsing** for every script (the common flags live
  in one place).
- **Rich `--help`** — the shared flags show up in each script's help.
- **Dup discovery** — if a script still declares its own copy of a
  shared flag, argparse raises `conflicting option string` at parser
  construction. That noise *is* the audit: most "script-specific" flags
  turned out to be stale duplicates of common ones, and were deleted.

## Converting a script

1. Import `common_parser` alongside `get_env_config`.
2. Add `parents=[common_parser()]` to the `ArgumentParser(...)`.
3. `parse_known_args()` → `parse_args()`; `get_env_config()` →
   `get_env_config(cli_args=args)`.
4. If `get_env_config()` ran *before* the parser was built, move it to
   *after* `parse_args()` (it now needs the namespace).
5. Run the script once. Delete any local `add_argument` that raises
   `conflicting option string` — it's a duplicate of a common flag.

### Per-command flags that aren't really common

A flag used by only one command does **not** belong in `common_parser()`.
`--pattern`/`--couchdb-pattern` were single-command attachment-name globs
masquerading as common flags; they were moved out to a per-command
`--attachment-pattern`. Don't add single-use flags to the common parser.

## The passthrough escape hatch

A script that genuinely forwards extras (e.g. a subprocess orchestrator)
keeps `parse_known_args()` and marks it:

```python
# pragma: argparse-passthrough -- <why this one can't be strict>
args, extras = parser.parse_known_args(main_argv)
```

Current holders:

- `manage_experiment.py` — subcommand parser plus a separate
  `get_env_config()` that reads common flags straight from `sys.argv`;
  splitting `--` passthrough to forwarded subprocess steps. Strict
  parsing would reject the common flags it must let through.
- `env_config.get_env_config()` — the standalone fallback for callers
  that don't pass a namespace; the *caller's* parser does the strict
  parse.

Keep the wording exactly `argparse-passthrough` — the hygiene test keys
on it.

## Enforcement

`bin/argparse_strict_test.py` has two guards:

- **static** (`test_no_unpragmaed_parse_known_args`) — AST-walks every
  `bin/` module; any `parse_known_args` *call* (comments ignored)
  without the pragma fails. Runs in milliseconds.
- **behavioral** (`test_cli_rejects_unknown_flag`) — launches every
  strict CLI with a bogus flag concurrently and asserts exit 2. Lazy ML
  imports keep argparse first, so it finishes in seconds; timeouts and
  unmet import deps are skipped, not failed.

## Out of scope

- Migrating to click / typer / fire. `parse_args()` does the job.
- Auto-suggesting "did you mean --X?" — argparse doesn't do this
  natively; nice-to-have, not required.
