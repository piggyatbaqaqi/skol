"""Tests for replicate_experiment pure helpers.

Covers the three functions that don't touch CouchDB or the network:
``databases_for_experiment``, ``build_couchdb_url``, and the
``_build_replicate_body`` body shape.

The network-touching ``replicate()`` and the ``main()`` orchestration
are exercised end-to-end against a real CouchDB pair when run by
hand; no fixture for that here.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replicate_experiment import (
    _build_replicate_body,
    build_couchdb_url,
    databases_for_experiment,
)


class TestDatabasesForExperiment(unittest.TestCase):
    """Project an experiment doc down to the list of DB names that
    must travel with it."""

    def test_skol_experiments_always_first(self):
        result = databases_for_experiment(
            {'databases': {'ingest': 'skol_dev'}},
        )
        self.assertEqual(result[0], 'skol_experiments')

    def test_v3_hand_shape(self):
        """Matches the production_v3_hand databases.* block we
        actually have in CouchDB."""
        exp = {
            '_id': 'production_v3_hand',
            'databases': {
                'ingest': 'skol_dev',
                'training': 'skol_training_v2_no_golden',
                'treatments': 'skol_treatments_v3_dev',
                'treatments_full': 'skol_treatments_full_v3_dev',
                'annotations': 'skol_exp_production_v3_hand_ann',
                'golden': 'skol_golden_v2',
                'golden_ann': 'skol_golden_ann_hand_v2',
            },
        }
        result = databases_for_experiment(exp)
        self.assertEqual(result[0], 'skol_experiments')
        self.assertIn('skol_dev', result)
        self.assertIn('skol_treatments_v3_dev', result)
        self.assertIn('skol_treatments_full_v3_dev', result)
        self.assertIn('skol_exp_production_v3_hand_ann', result)
        self.assertIn('skol_training_v2_no_golden', result)
        # Golden / golden_ann are excluded by default.
        self.assertNotIn('skol_golden_v2', result)
        self.assertNotIn('skol_golden_ann_hand_v2', result)

    def test_include_golden_adds_them(self):
        exp = {
            'databases': {
                'ingest': 'skol_dev',
                'golden': 'skol_golden_v2',
                'golden_ann': 'skol_golden_ann_hand_v2',
            },
        }
        result = databases_for_experiment(exp, include_golden=True)
        self.assertIn('skol_golden_v2', result)
        self.assertIn('skol_golden_ann_hand_v2', result)

    def test_duplicate_db_names_collapsed(self):
        """If two ``databases.*`` keys point at the same DB (e.g.,
        an old v1 experiment reused ``skol_treatments_dev`` for both
        ``treatments`` and ``treatments_full``), the result has it
        only once."""
        exp = {
            'databases': {
                'treatments': 'skol_treatments_dev',
                'treatments_full': 'skol_treatments_dev',
            },
        }
        result = databases_for_experiment(exp)
        self.assertEqual(result.count('skol_treatments_dev'), 1)

    def test_empty_or_missing_databases_block(self):
        """Bare experiment doc still yields the registry DB."""
        self.assertEqual(databases_for_experiment({}), ['skol_experiments'])
        self.assertEqual(
            databases_for_experiment({'databases': {}}),
            ['skol_experiments'],
        )

    def test_skips_empty_string_values(self):
        """Some experiment docs carry an explicit empty string for
        unused fields (e.g., ``"annotations": ""`` in v1).  Those
        must not appear in the replication target list."""
        exp = {
            'databases': {
                'ingest': 'skol_dev',
                'annotations': '',
                'treatments_full': None,
            },
        }
        result = databases_for_experiment(exp)
        self.assertEqual(result, ['skol_experiments', 'skol_dev'])


class TestBuildCouchdbUrl(unittest.TestCase):
    """Bare CouchDB base URL — no credential embedding (credentials
    travel separately in the replicate body's structured ``auth``
    block, since CouchDB's replicator doesn't percent-decode userinfo
    before sending Basic Auth)."""

    def test_default_port_and_scheme(self):
        self.assertEqual(
            build_couchdb_url(host='10.42.0.99'),
            'http://10.42.0.99:5984',
        )

    def test_custom_port_and_scheme(self):
        self.assertEqual(
            build_couchdb_url(
                host='prod.example', port=6984, scheme='https',
            ),
            'https://prod.example:6984',
        )


class TestBuildReplicateBody(unittest.TestCase):
    """Shape of the JSON body sent to ``POST /_replicate``.

    Source and target are emitted as bare URL strings when no auth is
    supplied, and as the structured ``{"url": ..., "auth": {"basic":
    {...}}}`` form when auth is supplied — this avoids the CouchDB
    replicator bug where percent-encoded ``@`` in URL userinfo is
    forwarded verbatim into the Basic Auth header."""

    def test_one_shot_body_no_auth(self):
        body = _build_replicate_body(
            'http://localhost:5984/src_db',
            'http://10.42.0.99:5984/src_db',
        )
        self.assertEqual(body['source'], 'http://localhost:5984/src_db')
        self.assertEqual(body['target'], 'http://10.42.0.99:5984/src_db')
        self.assertTrue(body['create_target'])
        self.assertNotIn('continuous', body)

    def test_continuous_body_sets_flag(self):
        body = _build_replicate_body(
            'http://localhost:5984/x', 'http://h:5984/x', continuous=True,
        )
        self.assertTrue(body['continuous'])

    def test_source_auth_wraps_as_structured_form(self):
        body = _build_replicate_body(
            'http://localhost:5984/db',
            'http://h:5984/db',
            source_auth=('admin', 'zd@GjUh77@5BHDQ'),
        )
        self.assertEqual(body['source'], {
            'url': 'http://localhost:5984/db',
            'auth': {'basic': {
                'username': 'admin',
                'password': 'zd@GjUh77@5BHDQ',
            }},
        })
        # Target stays a bare string — no auth supplied for it.
        self.assertEqual(body['target'], 'http://h:5984/db')

    def test_target_auth_wraps_as_structured_form(self):
        body = _build_replicate_body(
            'http://localhost:5984/db',
            'http://h:5984/db',
            target_auth=('admin', 'greatlyimprovedpassword'),
        )
        self.assertEqual(body['target'], {
            'url': 'http://h:5984/db',
            'auth': {'basic': {
                'username': 'admin',
                'password': 'greatlyimprovedpassword',
            }},
        })
        self.assertEqual(body['source'], 'http://localhost:5984/db')

    def test_both_sides_authed(self):
        body = _build_replicate_body(
            'http://localhost:5984/db',
            'http://h:5984/db',
            source_auth=('s_user', 's_pass'),
            target_auth=('t_user', 't_pass'),
        )
        self.assertEqual(
            body['source']['auth']['basic']['username'], 's_user',
        )
        self.assertEqual(
            body['target']['auth']['basic']['username'], 't_user',
        )

    def test_password_with_at_signs_passes_through_verbatim(self):
        """Regression: a ``@``-laden password must reach CouchDB
        unencoded in the structured ``auth`` block (previously it was
        being percent-encoded into the URL userinfo, which CouchDB's
        replicator forwarded verbatim into Basic Auth — the bug this
        whole shape change is fixing)."""
        body = _build_replicate_body(
            'http://localhost:5984/db', 'http://h:5984/db',
            source_auth=('admin', 'zd@GjUh77@5BHDQ'),
        )
        self.assertEqual(
            body['source']['auth']['basic']['password'],
            'zd@GjUh77@5BHDQ',
        )

    def test_empty_auth_tuple_treated_as_no_auth(self):
        """A ``('', '')`` tuple means admin-party; fall back to the
        bare-string URL form rather than emitting empty creds."""
        body = _build_replicate_body(
            'http://localhost:5984/db', 'http://h:5984/db',
            source_auth=('', ''), target_auth=('', ''),
        )
        self.assertEqual(body['source'], 'http://localhost:5984/db')
        self.assertEqual(body['target'], 'http://h:5984/db')

    def test_none_auth_treated_as_no_auth(self):
        body = _build_replicate_body(
            'http://localhost:5984/db', 'http://h:5984/db',
            source_auth=None, target_auth=None,
        )
        self.assertEqual(body['source'], 'http://localhost:5984/db')
        self.assertEqual(body['target'], 'http://h:5984/db')


# ---------------------------------------------------------------------------
# Endpoint shortcuts — same NAME-based credential resolution as
# bin/replicate_dbs
# ---------------------------------------------------------------------------


class TestResolveTargetEndpoint(unittest.TestCase):
    """``--target NAME`` resolves credentials via the same env-var
    convention as ``bin/replicate_dbs.resolve_endpoint``:
    ``<NAME>_COUCHDB_URL`` / ``_USER`` / ``_PASSWORD``.

    Falls back to the legacy ``--target-host`` / ``--target-port``
    / ``--target-scheme`` / ``--target-user`` / ``--target-pass``
    flags when the new ``--target NAME`` is not provided."""

    def _args(
        self, *,
        target: Optional[str] = None,
        target_host: Optional[str] = None,
        target_port: int = 5984,
        target_scheme: str = 'http',
        target_user: Optional[str] = None,
        target_pass: Optional[str] = None,
    ) -> Any:
        import argparse
        return argparse.Namespace(
            target=target,
            target_host=target_host,
            target_port=target_port,
            target_scheme=target_scheme,
            target_user=target_user,
            target_pass=target_pass,
        )

    def test_target_name_resolves_via_env_vars(self) -> None:
        from replicate_experiment import _resolve_target_endpoint
        env = {
            'PROD_COUCHDB_URL':      'https://prod.example.com:5984',
            'PROD_COUCHDB_USER':     'prod_admin',
            'PROD_COUCHDB_PASSWORD': 'prod_secret',
        }
        ep = _resolve_target_endpoint(self._args(target='prod'), env)
        self.assertEqual(ep.url, 'https://prod.example.com:5984')
        self.assertEqual(ep.username, 'prod_admin')
        self.assertEqual(ep.password, 'prod_secret')

    def test_target_local_alias_uses_unprefixed_vars(self) -> None:
        """The ``local`` / ``default`` aliases (matching the
        replicate_dbs convention) resolve via the bare
        ``COUCHDB_URL`` triple."""
        from replicate_experiment import _resolve_target_endpoint
        env = {
            'COUCHDB_URL':      'http://localhost:5984',
            'COUCHDB_USER':     'admin',
            'COUCHDB_PASSWORD': 'localpass',
        }
        ep = _resolve_target_endpoint(self._args(target='local'), env)
        self.assertEqual(ep.url, 'http://localhost:5984')
        self.assertEqual(ep.username, 'admin')
        self.assertEqual(ep.password, 'localpass')

    def test_legacy_target_host_still_works(self) -> None:
        """No ``--target NAME`` provided ⇒ assemble from the legacy
        ``--target-host`` / ``-port`` / ``-scheme`` / ``-user`` /
        ``-pass`` flags.  Preserves backward compat with any
        existing cron entries or scripts."""
        from replicate_experiment import _resolve_target_endpoint
        args = self._args(
            target_host='10.42.0.99',
            target_port=6984,
            target_scheme='https',
            target_user='admin',
            target_pass='legacy_pw',
        )
        ep = _resolve_target_endpoint(args, env={})
        self.assertEqual(ep.url, 'https://10.42.0.99:6984')
        self.assertEqual(ep.username, 'admin')
        self.assertEqual(ep.password, 'legacy_pw')

    def test_new_target_name_wins_over_legacy_flags(self) -> None:
        """When both ``--target NAME`` and the legacy flags are
        present, the new shortcut takes precedence.  Avoids the
        worst-of-both-worlds where an operator passes the new flag
        but a stale legacy flag silently overrides it."""
        from replicate_experiment import _resolve_target_endpoint
        env = {
            'PROD_COUCHDB_URL':      'https://prod.example.com:5984',
            'PROD_COUCHDB_USER':     'prod_admin',
            'PROD_COUCHDB_PASSWORD': 'prod_secret',
        }
        args = self._args(
            target='prod',
            target_host='ignore_me.example.com',
            target_user='legacy',
        )
        ep = _resolve_target_endpoint(args, env)
        self.assertEqual(ep.url, 'https://prod.example.com:5984')
        self.assertEqual(ep.username, 'prod_admin')

    def test_no_target_at_all_raises(self) -> None:
        """Neither ``--target NAME`` nor ``--target-host`` ⇒ user
        error.  Surface it loudly rather than silently picking
        defaults."""
        from replicate_experiment import _resolve_target_endpoint
        with self.assertRaises(ValueError):
            _resolve_target_endpoint(self._args(), env={})

    def test_target_name_missing_url_env_raises(self) -> None:
        """``--target prod`` with no ``PROD_COUCHDB_URL`` env var
        ⇒ ValueError (the resolve_endpoint contract from
        replicate_dbs)."""
        from replicate_experiment import _resolve_target_endpoint
        with self.assertRaises(ValueError):
            _resolve_target_endpoint(
                self._args(target='prod'), env={},
            )


class TestResolveSourceEndpoint(unittest.TestCase):
    """``--source NAME`` (default ``local``) follows the same
    naming convention.  Existing operational reality is "source
    is always the local CouchDB the script runs on", so default
    behaviour resolves to ``local``."""

    def test_default_source_is_local(self) -> None:
        """No ``--source`` argument ⇒ resolve as ``local``."""
        import argparse
        from replicate_experiment import _resolve_source_endpoint
        env = {
            'COUCHDB_URL':      'http://localhost:5984',
            'COUCHDB_USER':     'admin',
            'COUCHDB_PASSWORD': 'localpass',
        }
        ep = _resolve_source_endpoint(
            argparse.Namespace(source=None), env,
        )
        self.assertEqual(ep.url, 'http://localhost:5984')
        self.assertEqual(ep.username, 'admin')

    def test_named_source_resolves_via_env(self) -> None:
        """``--source skol`` resolves ``SKOL_COUCHDB_*``."""
        import argparse
        from replicate_experiment import _resolve_source_endpoint
        env = {
            'SKOL_COUCHDB_URL':      'https://synoptickeyof.life:5984',
            'SKOL_COUCHDB_USER':     'admin',
            'SKOL_COUCHDB_PASSWORD': 'prod_pw',
        }
        ep = _resolve_source_endpoint(
            argparse.Namespace(source='skol'), env,
        )
        self.assertEqual(ep.url, 'https://synoptickeyof.life:5984')
        self.assertEqual(ep.password, 'prod_pw')


if __name__ == '__main__':
    unittest.main()
