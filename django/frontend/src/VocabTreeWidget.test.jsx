/**
 * Tests for VocabTreeWidget: Proposal 1 of docs/plans/vocab-tree-narrowing.md.
 *
 * - The widget waits for a build it started: the build POST answers 202
 *   "building", and the widget polls GET build/ every 2 s, for up to
 *   3 minutes, instead of reporting a failure.
 * - Options carry their document counts, "Pileus (3,097)", in the order
 *   the server sends them.
 * - Counts never reach the description text.
 */
import {
  act, fireEvent, render, screen, waitFor,
} from '@testing-library/react';
import selectEvent from 'react-select-event';
import VocabTreeWidget from './VocabTreeWidget';

const TOP = {
  path: null,
  children: ['stipe', 'pileus'],
  counts: { stipe: 7, pileus: 3097 },
  count: 2,
  has_grandchildren: { stipe: false, pileus: true },
  is_leaf: false,
};
const PILEUS = {
  path: 'pileus',
  children: ['color'],
  counts: { color: 2810 },
  count: 1,
  has_grandchildren: { color: true },
  is_leaf: false,
};
const COLOR = {
  path: 'pileus.color',
  children: ['brown'],
  counts: { brown: 912 },
  count: 1,
  has_grandchildren: { brown: false },
  is_leaf: false,
};

function reply(status, body) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  });
}

/**
 * A fetch mock that routes by method and path.  `routes` maps
 * "GET children/", "POST build/" or "GET build/" to a function of the
 * call number (0, 1, ...) returning [status, body].
 */
function mockFetch(routes) {
  const calls = {};
  global.fetch = jest.fn((url, options = {}) => {
    const method = (options.method || 'GET').toUpperCase();
    const path = String(url).replace(/^.*\/vocab-tree\//, '').split('?')[0];
    const key = `${method} ${path}`;
    const n = calls[key] || 0;
    calls[key] = n + 1;
    const route = routes[key];
    if (!route) throw new Error(`unexpected fetch: ${key}`);
    const [status, body] = route(n);
    return reply(status, body);
  });
  return calls;
}

/** Children by path; the top level fails until `topFails` calls are used. */
function childrenRoute(topFails = 0) {
  let top = 0;
  return () => {
    const url = global.fetch.mock.calls.at(-1)[0];
    const path = new URL(url, 'http://x').searchParams.get('path') || '';
    if (path === '') {
      top += 1;
      if (top <= topFails) {
        return [500, { error: 'Vocabulary counts missing; rebuild' }];
      }
      return [200, TOP];
    }
    return [200, { pileus: PILEUS, 'pileus.color': COLOR }[path]];
  };
}

async function advance(ms) {
  await act(async () => {
    jest.advanceTimersByTime(ms);
  });
}

/** The dropdown for menu level ``n``, once the widget has rendered it. */
async function nthCombobox(n) {
  await waitFor(() => {
    expect(screen.getAllByRole('combobox').length).toBeGreaterThan(n);
  });
  return screen.getAllByRole('combobox')[n];
}

function optionTexts() {
  return screen.getAllByRole('option').map((o) => o.textContent);
}

afterEach(() => {
  jest.useRealTimers();
  delete global.fetch;
});

describe('waiting for a build the widget started', () => {
  test('polls GET build/ until the tree exists, then loads it', async () => {
    jest.useFakeTimers();
    const calls = mockFetch({
      'GET children/': childrenRoute(1),
      'POST build/': () => [202, { status: 'building' }],
      'GET build/': (n) => (n < 2
        ? [200, { status: 'not_found' }]
        : [200, { status: 'exists' }]),
    });
    render(<VocabTreeWidget />);
    await advance(0);
    for (let i = 0; i < 3; i += 1) {
      expect(screen.queryByText(/failed/i)).toBeNull();
      await advance(2000);
    }
    expect(calls['GET build/']).toBe(3);
    selectEvent.openMenu(screen.getAllByRole('combobox')[0]);
    expect(optionTexts()).toEqual(['Stipe (7)', 'Pileus (3,097)']);
    expect(screen.queryByText(/failed/i)).toBeNull();
  });

  test('gives up after 3 minutes with a notice, not an error', async () => {
    jest.useFakeTimers();
    const calls = mockFetch({
      'GET children/': childrenRoute(Infinity),
      'POST build/': () => [202, { status: 'building' }],
      'GET build/': () => [200, { status: 'not_found' }],
    });
    render(<VocabTreeWidget />);
    await advance(0);
    for (let t = 0; t < 180000; t += 2000) {
      await advance(2000);
    }
    expect(screen.getByText(/still being rebuilt/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    expect(screen.queryByText(/failed/i)).toBeNull();
    // Every 2 s for 3 minutes, and then no more.
    const polls = calls['GET build/'];
    expect(polls).toBeGreaterThanOrEqual(89);
    expect(polls).toBeLessThanOrEqual(91);
    await advance(20000);
    expect(calls['GET build/']).toBe(polls);
  });
});

describe('signed-in users', () => {
  // DRF rejects a signed-in user's POST without the CSRF header (403);
  // the widget's build request had none.
  test('the build POST sends the CSRF token', async () => {
    document.cookie = 'csrftoken=abc123';
    mockFetch({
      'GET children/': childrenRoute(1),
      'POST build/': () => [200, { status: 'exists' }],
    });
    render(<VocabTreeWidget />);
    await screen.findByRole('combobox');
    const post = global.fetch.mock.calls.find(
      ([, options]) => options && options.method === 'POST');
    expect(post[1].headers['X-CSRFToken']).toBe('abc123');
    expect(post[1].credentials).toBe('same-origin');
    document.cookie = 'csrftoken=; expires=Thu, 01 Jan 1970 00:00:00 GMT';
  });
});

describe('document counts', () => {
  test('labels options with counts, in the order the server sends', async () => {
    mockFetch({ 'GET children/': childrenRoute(0) });
    render(<VocabTreeWidget />);
    selectEvent.openMenu(await screen.findByRole('combobox'));
    expect(optionTexts()).toEqual(['Stipe (7)', 'Pileus (3,097)']);
  });

  test('keeps counts out of the description', async () => {
    mockFetch({ 'GET children/': childrenRoute(0) });
    const onAdd = jest.fn();
    render(<VocabTreeWidget onAddToDescription={onAdd} />);
    await selectEvent.select(await screen.findByRole('combobox'),
      'Pileus (3,097)');
    await selectEvent.select(await nthCombobox(1), 'Color (2,810)');
    await selectEvent.select(await nthCombobox(2), 'Brown (912)');
    fireEvent.click(screen.getByRole('button', { name: /add to description/i }));
    expect(onAdd).toHaveBeenCalledWith('pileus color brown; ');
  });
});
