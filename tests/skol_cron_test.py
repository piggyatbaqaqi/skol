"""Tests for debian/skol.cron: production's experiment-dependent jobs
run the ``production`` record's pipeline.

docs/plans/experiment-record-canonical.md, Step 3: every such job is
``manage_experiment runstep production <step>``, so the record's
pipeline (not the crontab) decides which script runs with which flags.

Run with: python -m pytest tests/skol_cron_test.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'bin'))

from pipelines import load  # type: ignore[import]  # noqa: E402

CRON = ROOT / 'debian' / 'skol.cron'

# The scripts a pipeline step runs; cron must not call them directly.
STEP_SCRIPTS = {
    'train_classifier', 'predict_classifier', 'predict_v4',
    'extract_treatments_to_couchdb', 'assess_treatments',
    'embed_treatments', 'treatments_to_json', 'build_vocab_tree',
    'canonicalize_json',
    'build_sources_stats',
}

# (schedule, step) for `manage_experiment runstep production <step>`.
EXPECTED = [
    ('0 0 * * *', 'predict'),
    ('0 4 * * *', 'extract_treatments'),
    ('0 5 * * *', 'assess_treatments'),
    ('0 6 * * *', 'embed'),
    ('0 7 * * *', 'build_sources_stats'),
    ('0 22 * * *', 'treatments_to_json'),
    ('0 22 * * 0,2,4,6', 'build_vocab'),
]


# Each scheduled experiment's pipeline.  The records live in CouchDB,
# so this mirrors them; keep it in step with the deployed records.
PIPELINES: Dict[str, str] = {
    'production': 'v4_crf',
    'production_v3_hand': 'v3_logistic',
}


class CronLine(NamedTuple):
    lineno: int
    schedule: str
    user: str
    command: str


def cron_lines() -> List[CronLine]:
    """The job lines: five schedule fields, a user, a command."""
    jobs = []
    for n, raw in enumerate(CRON.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#') or re.match(r'^\w+=', line):
            continue
        fields = line.split(None, 6)
        jobs.append(CronLine(n, ' '.join(fields[:5]), fields[5], fields[6]))
    return jobs


def runsteps(experiment: str) -> List[Tuple[str, str]]:
    pat = re.compile(
        rf'/bin/manage_experiment runstep {re.escape(experiment)} (\S+)')
    out = []
    for job in cron_lines():
        # Every runstep on the line: a job may chain several with &&.
        for m in pat.finditer(job.command):
            out.append((job.schedule, m.group(1)))
    return out


class TestCronParses:
    def test_job_lines_found(self) -> None:
        assert len(cron_lines()) > 20

    def test_v3_hand_runsteps_seen(self) -> None:
        """Sanity check on the parser: the production_v3_hand jobs
        already run through runstep."""
        assert ('0 12 * * *', 'predict') in runsteps('production_v3_hand')


class TestProductionRunsteps:
    @pytest.mark.parametrize('schedule,step', EXPECTED)
    def test_step_scheduled(self, schedule: str, step: str) -> None:
        assert (schedule, step) in runsteps('production')


class TestProductionStepsExist:
    def test_every_production_step_is_a_v4_crf_step(self) -> None:
        """Passes vacuously today; once the runsteps exist, each must
        name a step of the deployed pipeline."""
        names = {s.name for s in load('v4_crf')}
        assert {step for _, step in runsteps('production')} <= names


class TestExpectedStepsExist:
    def test_expected_steps_are_v4_crf_steps(self) -> None:
        names = {s.name for s in load('v4_crf')}
        assert {step for _, step in EXPECTED} <= names


class TestNoDirectStepScripts:
    def test_no_job_runs_a_step_script_directly(self) -> None:
        direct = [
            (job.lineno, script)
            for job in cron_lines()
            for script in STEP_SCRIPTS
            if re.search(rf'/bin/{script}(\s|$)', job.command)
        ]
        assert direct == []


def _time_and_days(schedule: str) -> Tuple[Tuple[int, int], str]:
    """((hour, minute), 'dom mon dow') for a fixed-time schedule."""
    minute, hour, rest = schedule.split(' ', 2)
    return (int(hour), int(minute)), rest


class TestAssessAfterEveryExtract:
    """Every ``extract_treatments`` job is followed, later the same
    day(s), by ``assess_treatments`` for the same experiment -- when
    that experiment's pipeline has the step."""

    def test_assess_follows_each_extract(self) -> None:
        missing = []
        for experiment, pipeline in PIPELINES.items():
            if 'assess_treatments' not in {s.name for s in load(pipeline)}:
                continue
            steps = runsteps(experiment)
            assess = [_time_and_days(sch) for sch, step in steps
                      if step == 'assess_treatments']
            for sch, step in steps:
                if step != 'extract_treatments':
                    continue
                when, days = _time_and_days(sch)
                if not any(d == days and t > when for t, d in assess):
                    missing.append((experiment, sch))
        assert missing == []

    def test_production_is_covered(self) -> None:
        """Guard against the rule passing vacuously."""
        assert ('0 4 * * *', 'extract_treatments') in runsteps('production')


class TestPipelinesTable:
    def test_names_real_pipelines(self) -> None:
        for pipeline in PIPELINES.values():
            assert load(pipeline)

    def test_v3_hand_exempt_for_now(self) -> None:
        """v3_logistic has no assess step; if one is added, cron must
        gain a production_v3_hand assess line (the rule above)."""
        names = {s.name for s in load('v3_logistic')}
        assert 'assess_treatments' not in names


class TestCanonicalizeBeforeBuildVocab:
    """docs/plans/vocab-tree-narrowing.md, Decisions for P2a: the menu
    build refuses while documents are stale, so each ``build_vocab`` job
    runs ``canonicalize_json`` first, in the same line, chained with
    ``&&`` so a failed canonicalization skips the build."""

    def build_vocab_jobs(self) -> List[Tuple[CronLine, str]]:
        jobs = []
        for job in cron_lines():
            for m in re.finditer(
                    r'/bin/manage_experiment runstep (\S+) build_vocab\b',
                    job.command):
                jobs.append((job, m.group(1)))
        return jobs

    def test_production_build_vocab_is_seen(self) -> None:
        """Guard against the rule passing vacuously."""
        assert 'production' in {exp for _, exp in self.build_vocab_jobs()}

    def test_each_build_vocab_canonicalizes_first(self) -> None:
        missing = []
        for job, exp in self.build_vocab_jobs():
            pat = (rf'/bin/manage_experiment runstep {re.escape(exp)} '
                   rf'canonicalize_json\b.*?\s&&\s.*/bin/manage_experiment '
                   rf'runstep {re.escape(exp)} build_vocab\b')
            if not re.search(pat, job.command):
                missing.append((job.lineno, exp))
        assert missing == []

    def test_canonicalize_json_is_a_v4_crf_step(self) -> None:
        assert 'canonicalize_json' in {s.name for s in load('v4_crf')}
