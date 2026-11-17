/**
 * The Django CSRF token, for requests from signed-in users.
 *
 * Django REST Framework's SessionAuthentication rejects an unsafe
 * request (POST, PUT, ...) from a signed-in user without an
 * X-CSRFToken header matching the csrftoken cookie.
 *
 * Stub: the tests in csrf.test.js are written first.
 */
export function getCSRFToken() {
  throw new Error('not implemented');
}
