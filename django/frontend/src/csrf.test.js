/**
 * Tests for csrf.js.  `test.failing` is Jest's strict xfail; remove it
 * when getCSRFToken is implemented.
 */
import { getCSRFToken } from './csrf';

function setCookies(...cookies) {
  // jsdom keeps cookies across tests; expire the ones a test set.
  document.cookie.split(';').forEach((c) => {
    const name = c.split('=')[0].trim();
    if (name) document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
  });
  cookies.forEach((c) => { document.cookie = c; });
}

test.failing('reads the csrftoken cookie', () => {
  setCookies('csrftoken=abc123');
  expect(getCSRFToken()).toBe('abc123');
});

test.failing('finds it among other cookies', () => {
  setCookies('sessionid=s1', 'csrftoken=abc123', 'theme=dark');
  expect(getCSRFToken()).toBe('abc123');
});

test.failing('is empty without the cookie', () => {
  setCookies('xcsrftoken=nope');
  expect(getCSRFToken()).toBe('');
});
