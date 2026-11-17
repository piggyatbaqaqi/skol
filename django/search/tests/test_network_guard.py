"""
Tests for skolweb.test_runner: the Django test runner blocks outbound
network connections and fails any test that attempts one.

Without it, 25 tests sent real requests to the developer's local
CouchDB (some with mock credentials); two full runs within a few
minutes locked the CouchDB admin account (2026-11-17).

Run with: python manage.py test search.tests.test_network_guard
"""
import io
import socket
import unittest

from django.test import SimpleTestCase

from skolweb import test_runner


def _a_test() -> unittest.TestCase:
    """A test object to feed the result class; not collected itself."""
    return unittest.FunctionTestCase(lambda: None)


class TestNetworkGuard(SimpleTestCase):

    def test_inet_connect_is_blocked(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with self.assertRaises(test_runner.NetworkBlockedError):
                sock.connect(('127.0.0.1', 5984))
        finally:
            sock.close()
            # This test's own attempt must not fail it.
            test_runner.attempts.clear()

    def test_result_fails_a_test_that_connected(self) -> None:
        result = test_runner.NetworkGuardResult(io.StringIO(), True, 0)
        test = _a_test()
        result.startTest(test)
        test_runner.attempts.append("('127.0.0.1', 5984)")
        result.stopTest(test)
        self.assertEqual(len(result.failures), 1)
        self.assertIn('127.0.0.1', result.failures[0][1])
        self.assertEqual(test_runner.attempts, [])

    def test_result_passes_a_test_that_did_not(self) -> None:
        result = test_runner.NetworkGuardResult(io.StringIO(), True, 0)
        test = _a_test()
        result.startTest(test)
        result.stopTest(test)
        self.assertEqual(result.failures, [])

    def test_runner_is_configured(self) -> None:
        from django.conf import settings
        self.assertEqual(settings.TEST_RUNNER,
                         'skolweb.test_runner.NetworkGuardRunner')
