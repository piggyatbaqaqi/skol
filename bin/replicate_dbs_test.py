"""Tests for bin/replicate_dbs.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replicate_dbs import (  # type: ignore[import]  # noqa: E402
    Endpoint,
    build_replication_payload,
    recreate_target_db,
    replicate_chain,
    resolve_endpoint,
)


# ---------------------------------------------------------------------------
# Fake HTTP session
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body
        self.text = '' if body is None else str(body)

    def json(self) -> Any:
        return self._body


class FakeSession:
    """Records every HTTP call so tests can assert on them."""

    def __init__(self, default: Optional[FakeResponse] = None,
                 responses: Optional[Dict[str, FakeResponse]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.default = default or FakeResponse(200, {'ok': True})
        self.responses = responses or {}

    def _route(self, method: str, url: str, **kw: Any) -> FakeResponse:
        self.calls.append({'method': method, 'url': url, **kw})
        return self.responses.get(f'{method} {url}', self.default)

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._route('GET', url, **kw)

    def put(self, url: str, **kw: Any) -> FakeResponse:
        return self._route('PUT', url, **kw)

    def delete(self, url: str, **kw: Any) -> FakeResponse:
        return self._route('DELETE', url, **kw)

    def post(self, url: str, **kw: Any) -> FakeResponse:
        return self._route('POST', url, **kw)


# ---------------------------------------------------------------------------
# resolve_endpoint
# ---------------------------------------------------------------------------


class TestResolveEndpoint(unittest.TestCase):
    """Endpoint name -> (url, username, password) via env vars.

    Convention: <NAME>_COUCHDB_URL + <NAME>_COUCHDB_USER +
    <NAME>_COUCHDB_PASSWORD.  Matches the local-alias convention
    (COUCHDB_URL / COUCHDB_USER / COUCHDB_PASSWORD) and the rest
    of the project's env-var naming."""

    def test_named_endpoint_full(self):
        env = {
            'TSQALI_COUCHDB_URL': 'https://tsq.example:16984',
            'TSQALI_COUCHDB_USER': 'admin',
            'TSQALI_COUCHDB_PASSWORD': 'secret',
        }
        ep = resolve_endpoint('tsqali', env)
        self.assertEqual(ep.url, 'https://tsq.example:16984')
        self.assertEqual(ep.username, 'admin')
        self.assertEqual(ep.password, 'secret')
        self.assertEqual(ep.name, 'tsqali')

    def test_named_endpoint_default_username(self):
        env = {
            'SKOL_COUCHDB_URL': 'https://skol.example:6984',
            'SKOL_COUCHDB_PASSWORD': 'prod-secret',
        }
        ep = resolve_endpoint('skol', env)
        self.assertEqual(ep.url, 'https://skol.example:6984')
        self.assertEqual(ep.username, 'admin')  # default
        self.assertEqual(ep.password, 'prod-secret')

    def test_named_endpoint_picks_up_user_env_var(self):
        """Regression: the named-endpoint path used to look for
        ``_COUCHDB_USERNAME`` which never matched what
        ``/home/skol/.skol_env`` actually sets, so production
        replication silently fell back to 'admin' and 401'd."""
        env = {
            'SKOL_COUCHDB_URL': 'https://skol.example:6984',
            'SKOL_COUCHDB_USER': 'skol',
            'SKOL_COUCHDB_PASSWORD': 'prod-secret',
        }
        ep = resolve_endpoint('skol', env)
        self.assertEqual(ep.username, 'skol')

    def test_local_falls_back_to_default_couchdb_vars(self):
        env = {
            'COUCHDB_URL': 'http://localhost:5984',
            'COUCHDB_USER': 'admin',
            'COUCHDB_PASSWORD': 'localpw',
        }
        ep = resolve_endpoint('local', env)
        self.assertEqual(ep.url, 'http://localhost:5984')
        self.assertEqual(ep.password, 'localpw')

    def test_default_synonym(self):
        env = {
            'COUCHDB_URL': 'http://localhost:5984',
            'COUCHDB_PASSWORD': 'pw',
        }
        ep = resolve_endpoint('default', env)
        self.assertEqual(ep.url, 'http://localhost:5984')

    def test_missing_url_raises(self):
        env: Dict[str, str] = {}
        with self.assertRaises(ValueError):
            resolve_endpoint('nowhere', env)

    def test_url_trailing_slash_stripped(self):
        env = {
            'TSQALI_COUCHDB_URL': 'https://tsq.example:16984/',
            'TSQALI_COUCHDB_PASSWORD': 'x',
        }
        ep = resolve_endpoint('tsqali', env)
        self.assertEqual(ep.url, 'https://tsq.example:16984')


# ---------------------------------------------------------------------------
# build_replication_payload
# ---------------------------------------------------------------------------


class TestBuildReplicationPayload(unittest.TestCase):
    """Object-form auth (not URL-embedded) so passwords containing
    '@' or other URL-reserved chars don't trip the URL parser — same
    trade-off we made in the original puchpuchobs/tsqali curl chain."""

    SRC = Endpoint(
        name='local', url='http://localhost:5984',
        username='admin', password='zd@GjUh77@5BHDQ',
    )
    TGT = Endpoint(
        name='tsqali', url='https://tsq.example:16984',
        username='admin', password='tsq-pw',
    )

    def test_payload_has_object_form_auth(self):
        body = build_replication_payload(self.SRC, self.TGT, 'skol_dev')
        self.assertEqual(body['source']['url'],
                         'http://localhost:5984/skol_dev')
        self.assertEqual(
            body['source']['auth']['basic']['password'],
            'zd@GjUh77@5BHDQ',
        )
        self.assertEqual(body['target']['url'],
                         'https://tsq.example:16984/skol_dev')
        self.assertEqual(
            body['target']['auth']['basic']['password'], 'tsq-pw',
        )

    def test_default_creates_target(self):
        body = build_replication_payload(self.SRC, self.TGT, 'skol_dev')
        self.assertTrue(body.get('create_target'))

    def test_create_target_can_be_disabled(self):
        body = build_replication_payload(
            self.SRC, self.TGT, 'skol_dev', create_target=False,
        )
        self.assertFalse(body.get('create_target'))

    def test_default_leaves_use_bulk_get_unset(self):
        """When the caller doesn't ask, leave ``use_bulk_get`` out of
        the payload — CouchDB then uses its own default (``true`` on
        modern releases).  Don't surprise operators by forcing a
        slower fallback on every replication."""
        body = build_replication_payload(self.SRC, self.TGT, 'skol_dev')
        self.assertNotIn('use_bulk_get', body)

    def test_use_bulk_get_false_emits_payload_key(self):
        """``--no-bulk-get`` opt-out: when the remote's `_bulk_get`
        is broken (e.g. multipart corruption — see
        couch_replicator_api_wrap:bulk_get in the CouchDB log),
        the replicator falls back to per-doc _open_revs GETs."""
        body = build_replication_payload(
            self.SRC, self.TGT, 'skol_dev', use_bulk_get=False,
        )
        self.assertIs(body['use_bulk_get'], False)


# ---------------------------------------------------------------------------
# recreate_target_db
# ---------------------------------------------------------------------------


class TestRecreateTargetDb(unittest.TestCase):
    """--recreate-target: DELETE then PUT the target DB so the next
    replication writes into an empty database (the only way to make
    target *exactly* match source when target has extra docs)."""

    TGT = Endpoint(
        name='tsqali', url='https://tsq.example:16984',
        username='admin', password='pw',
    )

    def test_delete_then_put(self):
        sess = FakeSession()
        recreate_target_db(self.TGT, 'skol_dev', http=sess)
        methods = [c['method'] for c in sess.calls]
        urls = [c['url'] for c in sess.calls]
        self.assertEqual(methods, ['DELETE', 'PUT'])
        self.assertTrue(all(
            u == 'https://tsq.example:16984/skol_dev' for u in urls
        ))

    def test_delete_404_is_ignored(self):
        """A 404 on the initial DELETE just means the DB doesn't
        exist yet — that's fine, we'd PUT it anyway."""
        sess = FakeSession(responses={
            'DELETE https://tsq.example:16984/skol_dev':
                FakeResponse(404, {'error': 'not_found'}),
            'PUT https://tsq.example:16984/skol_dev':
                FakeResponse(201, {'ok': True}),
        })
        # Should not raise.
        recreate_target_db(self.TGT, 'skol_dev', http=sess)
        self.assertEqual([c['method'] for c in sess.calls],
                         ['DELETE', 'PUT'])

    def test_put_failure_raises(self):
        """A PUT that returns non-2xx is a real failure we must
        surface — caller can't continue without the empty target."""
        sess = FakeSession(responses={
            'DELETE https://tsq.example:16984/skol_dev':
                FakeResponse(200, {'ok': True}),
            'PUT https://tsq.example:16984/skol_dev':
                FakeResponse(403, 'forbidden'),
        })
        with self.assertRaises(Exception):
            recreate_target_db(self.TGT, 'skol_dev', http=sess)


# ---------------------------------------------------------------------------
# replicate_chain
# ---------------------------------------------------------------------------


class TestReplicateChain(unittest.TestCase):
    """The chain runs replication hop by hop.  Each hop POSTs to
    PREV's /_replicate endpoint so PREV pushes into NEXT — same
    direction the manual curl chain used (source initiates)."""

    LOCAL = Endpoint(
        name='local', url='http://localhost:5984',
        username='admin', password='lpw',
    )
    TSQALI = Endpoint(
        name='tsqali', url='https://tsq.example:16984',
        username='admin', password='tpw',
    )
    PROD = Endpoint(
        name='skol', url='https://skol.example:6984',
        username='admin', password='ppw',
    )

    def test_two_hop_chain_posts_to_each_source(self):
        sess = FakeSession()
        replicate_chain(
            [self.LOCAL, self.TSQALI, self.PROD],
            'skol_dev', http=sess,
        )
        posts = [c for c in sess.calls if c['method'] == 'POST']
        self.assertEqual(len(posts), 2)
        self.assertEqual(
            posts[0]['url'], 'http://localhost:5984/_replicate',
        )
        self.assertEqual(
            posts[1]['url'],
            'https://tsq.example:16984/_replicate',
        )

    def test_single_hop_one_post(self):
        sess = FakeSession()
        replicate_chain(
            [self.LOCAL, self.TSQALI],
            'skol_dev', http=sess,
        )
        posts = [c for c in sess.calls if c['method'] == 'POST']
        self.assertEqual(len(posts), 1)

    def test_dry_run_makes_no_http_calls(self):
        sess = FakeSession()
        replicate_chain(
            [self.LOCAL, self.TSQALI, self.PROD],
            'skol_dev', http=sess, dry_run=True,
        )
        self.assertEqual(sess.calls, [])

    def test_recreate_target_does_drop_create_then_replicate(self):
        sess = FakeSession()
        replicate_chain(
            [self.LOCAL, self.TSQALI],
            'skol_dev', http=sess, recreate=True,
        )
        methods = [c['method'] for c in sess.calls]
        # DELETE tsqali/skol_dev, PUT tsqali/skol_dev, POST .../_replicate
        self.assertEqual(methods, ['DELETE', 'PUT', 'POST'])

    def test_replication_body_carries_object_form_auth(self):
        """Sanity: the POST body we send is the same object-form
        payload build_replication_payload returns — so the @-in-pw
        protection actually reaches the wire."""
        sess = FakeSession()
        local_with_at = Endpoint(
            name='local', url='http://localhost:5984',
            username='admin', password='a@b@c',
        )
        replicate_chain(
            [local_with_at, self.TSQALI],
            'skol_dev', http=sess,
        )
        post = next(c for c in sess.calls if c['method'] == 'POST')
        body = post.get('json', {})
        self.assertEqual(
            body['source']['auth']['basic']['password'], 'a@b@c',
        )


if __name__ == '__main__':
    unittest.main()
