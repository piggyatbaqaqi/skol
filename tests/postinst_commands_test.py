"""The commands the deb puts on PATH, against the ``bin/`` symlinks and
the commands skol's own messages tell people to run.

``debian/postinst.template`` links each name in its ``for cmd in`` list
to ``with_skol`` in ``/opt/skol/bin``; a checkout gets the same names as
committed ``bin/<name> -> with_skol`` symlinks.  P2a shipped
``bin/canonicalize_json.py`` without either, so production answered
``canonicalize_json: command not found`` to the very command the
refusal message prints (Trello AZeJMCR3, 2026-11-22).

Run with: python -m pytest tests/postinst_commands_test.py -v
"""
import ast
import re
from pathlib import Path
from typing import Dict, Iterator, List, Set

import pytest

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / 'bin'
POSTINST = ROOT / 'debian' / 'postinst.template'

# Where hints that name a command live.
HINT_SOURCES = ('bin', 'treatments_to_structured')
# "run: canonicalize_json ...", "Run: manage_experiment ...",
# "re-pin with: canonicalize_json ...".
HINT = re.compile(r'\b(?:[Rr]un|with):\s+`?([a-z][a-z0-9_]*)\b')

# No exceptions (ruled 2026-11-22): every bin/ symlink is installed and
# every installed command is linked, even tools that make little sense
# on production.  An exception list would be one more thing to drift.


def postinst_lines() -> List[str]:
    """The raw lines of the postinst's ``for cmd in \\`` list."""
    lines = POSTINST.read_text().splitlines()
    start = lines.index('for cmd in \\') + 1
    end = lines.index('; do', start)
    return lines[start:end]


def postinst_commands() -> List[str]:
    """The names in the postinst's command list, in file order."""
    return [line.strip().rstrip('\\').strip() for line in postinst_lines()]


def symlink_commands() -> Set[str]:
    """Names committed as ``bin/<name> -> with_skol``."""
    return {p.name for p in BIN.iterdir()
            if p.is_symlink() and str(p.readlink()) == 'with_skol'}


def bin_scripts() -> Set[str]:
    """``bin/<name>.py`` programs, tests left out."""
    return {p.stem for p in BIN.glob('*.py')
            if not p.stem.endswith('_test')}


def _strings(tree: ast.AST) -> Iterator[str]:
    """Every string literal, adjacent literals joined as Python joins
    them; f-string fields become ``{}``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            yield ''.join(str(v.value) if isinstance(v, ast.Constant)
                          else '{}' for v in node.values)


def hinted_commands() -> Dict[str, Set[str]]:
    """Each ``bin/`` program a message tells the user to run, and the
    files whose messages name it."""
    scripts = bin_scripts()
    found: Dict[str, Set[str]] = {}
    for top in HINT_SOURCES:
        for path in sorted((ROOT / top).glob('*.py')):
            if path.stem.endswith('_test'):
                continue
            for text in _strings(ast.parse(path.read_text())):
                for match in HINT.finditer(text):
                    if match.group(1) in scripts:
                        found.setdefault(match.group(1), set()).add(
                            str(path.relative_to(ROOT)))
    return found


class TestHelpers:
    """The scans find what they are meant to (passes today)."""

    def test_postinst_list_found(self) -> None:
        assert 'manage_experiment' in postinst_commands()
        assert 'build_vocab_tree' in postinst_commands()

    def test_symlinks_found(self) -> None:
        assert 'build_vocab_tree' in symlink_commands()

    def test_hints_span_adjacent_literals(self) -> None:
        """build_vocab_tree's refusal splits its hint over two literals."""
        hints = hinted_commands()
        assert 'bin/build_vocab_tree.py' in hints['manage_experiment']
        assert 'bin/build_vocab_tree.py' in hints['canonicalize_json']


class TestCanonicalizeJsonIsACommand:
    """A. The P2a step is installed and linked."""

    @pytest.mark.xfail(strict=True,
                       reason='canonicalize_json is not in the postinst list')
    def test_installed(self) -> None:
        assert 'canonicalize_json' in postinst_commands()

    @pytest.mark.xfail(strict=True,
                       reason='no bin/canonicalize_json symlink')
    def test_linked(self) -> None:
        assert 'canonicalize_json' in symlink_commands()


class TestPostinstListForm:
    """B. One name per line, sorted, as the list's comment asks."""

    def test_one_name_per_line(self) -> None:
        for line in postinst_lines():
            assert re.fullmatch(r'    [a-z][a-z0-9_]* \\', line), line

    def test_sorted(self) -> None:
        names = postinst_commands()
        assert names == sorted(names)

    def test_no_duplicates(self) -> None:
        names = postinst_commands()
        assert len(names) == len(set(names))

    def test_each_name_is_a_script(self) -> None:
        """A name with no bin/<name>.py makes with_skol fail at run time."""
        missing = set(postinst_commands()) - bin_scripts()
        assert not missing, sorted(missing)


class TestHintedCommandsAreInstalled:
    """C. A message that says "run: X" names an installed command."""

    @pytest.mark.xfail(strict=True,
                       reason='canonicalize_json is named in hints but '
                              'not installed')
    def test_every_hinted_command_is_installed(self) -> None:
        missing = {name: sorted(files)
                   for name, files in hinted_commands().items()
                   if name not in postinst_commands()}
        assert not missing, missing


class TestListsMatch:
    """D. The postinst list and the bin/ symlinks name the same
    commands, with no exceptions."""

    @pytest.mark.xfail(
        strict=True,
        reason='drift.  Symlinked, not installed: '
               'assess_treatments, brat_export, brat_ingest, '
               'build_canonical_labels, '
               'llm_annotate_features, prepare_modal_models, '
               'report_round_review, select_for_annotation, '
               'train_crf_single, treatment_dossier, triage_treatments.  '
               'Installed, no symlink: jats_to_yedda, prod_smoke_check, '
               'watch_incremental')
    def test_lists_match(self) -> None:
        installed = set(postinst_commands())
        linked = symlink_commands()
        assert {'symlinked, not installed': sorted(linked - installed),
                'installed, no symlink': sorted(installed - linked)} == {
            'symlinked, not installed': [], 'installed, no symlink': []}
