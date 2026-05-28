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
    """URL-encode embedded credentials for CouchDB targets."""

    def test_no_auth_target(self):
        """An admin-party target (no creds) returns the bare URL."""
        self.assertEqual(
            build_couchdb_url(host='10.42.0.99'),
            'http://10.42.0.99:5984',
        )
        self.assertEqual(
            build_couchdb_url(host='10.42.0.99', user='', password=''),
            'http://10.42.0.99:5984',
        )

    def test_plain_credentials(self):
        self.assertEqual(
            build_couchdb_url(host='localhost', user='admin', password='secret'),
            'http://admin:secret@localhost:5984',
        )

    def test_password_with_at_sign_is_url_encoded(self):
        """The real local-CouchDB password ``zd@GjUh77@5BHDQ`` has two
        ``@`` characters.  Without encoding, urllib would parse the
        second ``@`` as the host delimiter and the URL would be
        unreachable.  ``%40`` is the canonical encoding."""
        result = build_couchdb_url(
            host='localhost', user='admin', password='zd@GjUh77@5BHDQ',
        )
        self.assertEqual(
            result, 'http://admin:zd%40GjUh77%405BHDQ@localhost:5984',
        )

    def test_password_with_other_special_characters(self):
        """Colons and slashes inside the password get encoded too."""
        result = build_couchdb_url(
            host='localhost', user='admin', password='a:b/c?d',
        )
        self.assertIn('a%3Ab%2Fc%3Fd', result)

    def test_custom_port_and_scheme(self):
        self.assertEqual(
            build_couchdb_url(
                host='prod.example', user='u', password='p',
                port=6984, scheme='https',
            ),
            'https://u:p@prod.example:6984',
        )

    def test_missing_one_credential_omits_both(self):
        """``user`` without ``password`` (or vice versa) is degenerate;
        fall back to the no-auth shape rather than build a malformed
        URL."""
        self.assertEqual(
            build_couchdb_url(host='h', user='admin', password=''),
            'http://h:5984',
        )
        self.assertEqual(
            build_couchdb_url(host='h', user='', password='secret'),
            'http://h:5984',
        )


class TestBuildReplicateBody(unittest.TestCase):
    """Shape of the JSON body sent to ``POST /_replicate``."""

    def test_one_shot_body(self):
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


if __name__ == '__main__':
    unittest.main()
