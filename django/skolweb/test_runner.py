"""Django test runner that blocks outbound network connections.

Tests must not reach real services.  Without this guard, 25 search tests
sent requests to the developer's local CouchDB, some with mock
credentials, and two full runs within a few minutes locked the CouchDB
admin account (2026-11-17).  The views swallowed the failures, so every
test still passed and nothing showed the leak.

While the runner is active, connecting an INET or INET6 socket raises
``NetworkBlockedError`` (a ``ConnectionRefusedError``, so code under
test sees an ordinary refused connection), and the test that tried it
fails, naming the addresses.  Unix sockets are unaffected.  The
Collection -> CouchDB sync (``COUCHDB_SYNC_ENABLED``) is turned off, so
saving a Collection in a test writes nothing to CouchDB.

Configured as ``TEST_RUNNER`` in skolweb/settings.py.
"""
import socket
import sys
import unittest
from typing import Any, Callable, List, Optional, Tuple, Type

from django.test.runner import DiscoverRunner

# Addresses connected to since the current test started.
attempts: List[str] = []

_BLOCKED_FAMILIES = (socket.AF_INET, socket.AF_INET6)
_saved: Optional[Tuple[Callable[..., Any], Callable[..., Any]]] = None


class NetworkBlockedError(ConnectionRefusedError):
    """Raised for an outbound connection attempted during a test."""


def _blocking(original: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: socket.socket, address: Any) -> Any:
        if self.family in _BLOCKED_FAMILIES:
            attempts.append(repr(address))
            raise NetworkBlockedError(
                f'outbound network blocked during tests: {address!r}')
        return original(self, address)
    return guarded


def install() -> None:
    """Block INET connections (idempotent)."""
    global _saved
    if _saved is not None:
        return
    _saved = (socket.socket.connect, socket.socket.connect_ex)
    socket.socket.connect = _blocking(_saved[0])  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocking(_saved[1])  # type: ignore[method-assign]


def uninstall() -> None:
    """Restore normal connections."""
    global _saved
    if _saved is None:
        return
    socket.socket.connect, socket.socket.connect_ex = _saved  # type: ignore[method-assign]
    _saved = None
    attempts.clear()


class NetworkGuardResult(unittest.TextTestResult):
    """Fails a test that attempted an outbound connection."""

    def startTest(self, test: unittest.TestCase) -> None:
        attempts.clear()
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:
        if attempts:
            targets = ', '.join(sorted(set(attempts)))
            attempts.clear()
            try:
                raise AssertionError(
                    f'test made outbound network connection(s): {targets}.  '
                    f'Patch the call (requests, get_experiment_config, ...) '
                    f'instead.')
            except AssertionError:
                self.addFailure(test, sys.exc_info())
        super().stopTest(test)


class NetworkGuardRunner(DiscoverRunner):
    """DiscoverRunner with outbound network blocked for every test."""

    def get_resultclass(self) -> Optional[Type[unittest.TextTestResult]]:
        base = super().get_resultclass()
        if base is None:
            return NetworkGuardResult
        return type(f'NetworkGuard{base.__name__}',
                    (NetworkGuardResult, base), {})

    def setup_test_environment(self, **kwargs: Any) -> None:
        super().setup_test_environment(**kwargs)
        from django.conf import settings
        self._sync_was = getattr(settings, 'COUCHDB_SYNC_ENABLED', True)
        settings.COUCHDB_SYNC_ENABLED = False
        install()

    def teardown_test_environment(self, **kwargs: Any) -> None:
        uninstall()
        from django.conf import settings
        settings.COUCHDB_SYNC_ENABLED = self._sync_was
        super().teardown_test_environment(**kwargs)
