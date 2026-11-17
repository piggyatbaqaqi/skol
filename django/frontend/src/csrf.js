/**
 * The Django CSRF token, for requests from signed-in users.
 *
 * Django REST Framework's SessionAuthentication rejects an unsafe
 * request (POST, PUT, ...) from a signed-in user without an
 * X-CSRFToken header matching the csrftoken cookie.
 *
 * Returns the cookie's value, or '' when there is none.
 */
export function getCSRFToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
  return match ? match[1] : '';
}
