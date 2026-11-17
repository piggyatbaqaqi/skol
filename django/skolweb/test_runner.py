"""Django test runner that blocks outbound network connections.

Stub: the tests in search/tests/test_network_guard.py are written first.
"""
from typing import List

from django.test.runner import DiscoverRunner

attempts: List[str] = []


class NetworkBlockedError(ConnectionRefusedError):
    """Raised for an outbound connection attempted during a test."""


class NetworkGuardResult:
    """Stub."""


class NetworkGuardRunner(DiscoverRunner):
    """Stub."""
