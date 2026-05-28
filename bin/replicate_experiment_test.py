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


if __name__ == '__main__':
    unittest.main()
