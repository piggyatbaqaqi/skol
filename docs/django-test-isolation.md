# Django tests are isolated from real services

`python manage.py test` (in `django/`) runs every test under
`skolweb.test_runner.NetworkGuardRunner`, configured as `TEST_RUNNER` in
`django/skolweb/settings.py`.  It gives two guarantees:

1. **No test opens a network connection.**  Connecting an INET or INET6
   socket raises `NetworkBlockedError`, and the test that tried fails,
   naming the address:

   ```
   AssertionError: test made outbound network connection(s):
   ('127.0.0.1', 5984).  Patch the call (requests,
   get_experiment_config, ...) instead.
   ```

   `NetworkBlockedError` is a `ConnectionRefusedError`, so the code under
   test sees an ordinary refused connection.  The test fails anyway, even
   when the code swallows the error, which most views do.  Unix sockets
   are not blocked.
2. **Saving a `Collection` writes nothing to CouchDB.**  The runner sets
   `COUCHDB_SYNC_ENABLED = False`, and the sync functions in
   `search/couchdb_sync.py` then return `False` without connecting.
   Those functions are `sync_collection_to_couchdb` and
   `delete_collection_from_couchdb`, called by the post_save and
   post_delete signals, by the comment "copy nomenclature" view and by
   `sync_all_collections`.

## Why

Found on 2026-11-17, during the vocab tree P1 hand checks:

* **157 tests reached real local services.** 156 connected to the local
  CouchDB (`127.0.0.1:5984`) and 1 to the local Redis (`127.0.0.1:6380`).
  The views and signals swallowed the failures, so every test passed and
  nothing showed the leak.
* **The Collection sync wrote test data into the local CouchDB.** It ran
  with real credentials.  By that date 5,568 test documents had built up
  in `skol_collections_dev`, with their history records in
  `skol_collections_history_dev`.  They were deleted on 2026-11-17 with
  `fixes/expunge_orphan_collections.py --apply` (Trello 9ZkM5Wuq):
  11,371 documents in all, 5,568 collections and 5,803 history
  records.  The dev database's 12 collections and their 32 history
  records were kept.
* **Mock credentials locked the CouchDB account.** The classifier-view
  tests mock `search.views.settings`, so they sent `MagicMock`
  credentials.  Two full runs within a few minutes locked the CouchDB
  `admin` account: "Account is temporarily locked due to multiple
  authentication failures".  While the lock lasted, the local dev
  server failed every CouchDB request.

Production was never affected: the tests only reached localhost.

The fix is Trello KosomUsu, commits `6265c40`, `219ea9d` and `3163bdc`.

## Writing tests that pass the guard

Patch the call at the boundary, so the test gets the answer it needs and
no socket opens:

| The code calls | Patch |
|---|---|
| `requests.get/post` in a view | `patch('search.views.requests.get', ...)`, returning a `MagicMock(status_code=..., json=...)` |
| `get_experiment_config` / `get_user_experiment` | `patch('search.views.get_experiment_config', return_value=None)` (or a record dict) |
| Redis through `get_redis_client` | `patch('search.views.get_redis_client', return_value=FakeRedis(...))`; see `search/tests/test_vocab_menus.py` |
| `get_couchdb_server` in a service | `patch('search.export_service.get_couchdb_server', return_value=server)` with a `MagicMock` server |
| The Collection sync itself | Nothing to do: it is off.  To test it, use `@override_settings(COUCHDB_SYNC_ENABLED=True)` and patch `search.couchdb_sync.get_couchdb_server` (see `search/tests/test_couchdb_sync_gate.py`) |

A patch that replaces a failing lookup should return what the failure
produced (`None`, a 404, an empty server), so the test's assertions
keep meaning what they meant.

## Turning the sync off elsewhere

`COUCHDB_SYNC_ENABLED` is read from the environment and defaults to on
(`0`, `false` or `no` turn it off).  A dev server that should not write
to the local CouchDB can run with `COUCHDB_SYNC_ENABLED=false`.  Search
then misses the collections that server creates.

## Scope

* The guard covers the Django suite (`manage.py test`).  The pytest
  suite at the repository root is not covered.
* Tests run under `--parallel` use Django's remote result class, which
  this runner does not wrap.  The socket block still applies there, but
  a leak would show only as the code's own error handling.
