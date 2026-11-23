/**
 * Tests for FeatureSelectionWidget's two new tabs, Text Splits and
 * Structured Splits: plan F3 of docs/plans/result-splitting-features.md.
 * The old tabs stay as they are (new tools, not replacements).
 *
 * test.failing marks tests that must fail until F3 is implemented, as
 * pytest's strict xfail does.
 */
import {
  act, fireEvent, render, screen, waitFor,
} from '@testing-library/react';
import FeatureSelectionWidget from './FeatureSelectionWidget';

jest.mock('./VocabTreeWidget', () => () => <div>vocabulary</div>);
jest.mock('./MetricsWidget', () => () => <div>metrics</div>);

const STRUCTURED = {
  n_results: 2,
  n_missing: 0,
  n_fallback: 0,
  canonical_version: null,
  parts: [{
    path: ['Pileus'],
    described: 2,
    score: 0.5,
    attributes: [{
      path: ['Pileus', 'Color'],
      score: 0.5,
      not_mentioned: 0,
      values: [
        { value: 'Brown', count: 1, others: 0.5,
          display_text: 'Pileus color brown' },
        { value: 'White', count: 1, others: 0.5,
          display_text: 'Pileus color white' },
      ],
    }],
    parts: [],
  }],
};
const TEXT = {
  n_results: 2,
  n_missing: 0,
  features: [{ text: 'clamp connections', count: 1, score: 0.5 }],
};

function reply(status, body) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  });
}

function mockFetch(byPath) {
  global.fetch = jest.fn((url) => {
    const path = String(url).replace(/^.*\/api\//, '');
    if (!(path in byPath)) throw new Error(`unexpected fetch: ${path}`);
    return reply(...byPath[path]);
  });
}

function searchDone(ids) {
  act(() => {
    document.dispatchEvent(new CustomEvent('search-results-ready',
      { detail: { treatment_ids: ids } }));
  });
}

function bodyOf(call) {
  return JSON.parse(call[1].body);
}

beforeEach(() => {
  mockFetch({
    'features/structured/': [200, STRUCTURED],
    'features/text/': [200, TEXT],
  });
});

afterEach(() => {
  document.body.innerHTML = '';
});

test.failing('the new tabs sit after the old feature tabs', () => {
  render(<FeatureSelectionWidget />);
  expect(screen.getAllByRole('tab').map(t => t.textContent)).toEqual([
    'Vocabulary', 'Text Features', 'JSON Features', 'Text Splits',
    'Structured Splits', 'Metrics',
  ]);
});

test.failing('arrow keys reach the new tabs', () => {
  render(<FeatureSelectionWidget />);
  const jsonTab = screen.getByRole('tab', { name: 'JSON Features' });
  fireEvent.click(jsonTab);
  fireEvent.keyDown(jsonTab, { key: 'ArrowRight' });
  expect(screen.getByRole('tab', { name: 'Text Splits' }))
    .toHaveAttribute('aria-selected', 'true');
});

test.failing('before a search, the tab asks for one and fetches nothing', () => {
  render(<FeatureSelectionWidget />);
  fireEvent.click(screen.getByRole('tab', { name: 'Structured Splits' }));
  expect(screen.getByRole('status')).toHaveTextContent(/run a search/i);
  expect(global.fetch).not.toHaveBeenCalled();
});

test.failing('Structured Splits posts the results and the top-N setting', async () => {
  document.body.innerHTML = '<input id="featureTopN" value="12">';
  render(<FeatureSelectionWidget />);
  searchDone(['a', 'b']);
  fireEvent.click(screen.getByRole('tab', { name: 'Structured Splits' }));
  await screen.findByRole('checkbox', { name: /Brown/ });
  const [url, options] = global.fetch.mock.calls[0];
  expect(url).toBe('/api/features/structured/');
  expect(options.method).toBe('POST');
  expect(bodyOf(global.fetch.mock.calls[0])).toEqual(
    { treatment_ids: ['a', 'b'], top_n: 12 });
});

test.failing('Text Splits posts to its own endpoint', async () => {
  render(<FeatureSelectionWidget />);
  searchDone(['a', 'b']);
  fireEvent.click(screen.getByRole('tab', { name: 'Text Splits' }));
  await screen.findByRole('checkbox', { name: 'clamp connections' });
  expect(global.fetch.mock.calls[0][0]).toBe('/api/features/text/');
  expect(bodyOf(global.fetch.mock.calls[0])).toEqual(
    { treatment_ids: ['a', 'b'], top_n: 30 });
});

test.failing('each tab fetches once per set of results', async () => {
  render(<FeatureSelectionWidget />);
  searchDone(['a', 'b']);
  const tab = screen.getByRole('tab', { name: 'Structured Splits' });
  fireEvent.click(tab);
  await screen.findByRole('checkbox', { name: /Brown/ });
  fireEvent.click(screen.getByRole('tab', { name: 'Vocabulary' }));
  fireEvent.click(tab);
  expect(global.fetch).toHaveBeenCalledTimes(1);
  searchDone(['c']);
  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
  expect(bodyOf(global.fetch.mock.calls[1]).treatment_ids).toEqual(['c']);
});

test.failing('a selection survives switching tabs', async () => {
  render(<FeatureSelectionWidget />);
  searchDone(['a', 'b']);
  fireEvent.click(screen.getByRole('tab', { name: 'Structured Splits' }));
  fireEvent.click(await screen.findByRole('checkbox', { name: /Brown/ }));
  fireEvent.click(screen.getByRole('tab', { name: 'Vocabulary' }));
  fireEvent.click(screen.getByRole('tab', { name: 'Structured Splits' }));
  expect(screen.getByRole('checkbox', { name: /Brown/ })).toBeChecked();
});

test.failing('adding from Structured Splits names its source', async () => {
  const onAdd = jest.fn();
  render(<FeatureSelectionWidget onAddToDescription={onAdd} />);
  searchDone(['a', 'b']);
  fireEvent.click(screen.getByRole('tab', { name: 'Structured Splits' }));
  fireEvent.click(await screen.findByRole('checkbox', { name: /Brown/ }));
  fireEvent.click(screen.getByRole('button', { name: /add to description/i }));
  expect(onAdd).toHaveBeenCalledWith('Pileus color brown; ',
                                     'structured-splits');
});

test.failing('adding from Text Splits names its source and fills the description', async () => {
  const onAdd = jest.fn();
  const ref = { current: document.createElement('textarea') };
  render(<FeatureSelectionWidget onAddToDescription={onAdd}
                                 descriptionRef={ref} />);
  searchDone(['a', 'b']);
  fireEvent.click(screen.getByRole('tab', { name: 'Text Splits' }));
  fireEvent.click(await screen.findByRole('checkbox',
                                          { name: 'clamp connections' }));
  fireEvent.click(screen.getByRole('button', { name: /add to description/i }));
  expect(onAdd).toHaveBeenCalledWith('clamp connections; ', 'text-splits');
  expect(ref.current.value).toContain('clamp connections;');
});

test.failing('a failed request shows the server error', async () => {
  mockFetch({ 'features/structured/': [500, { error: 'couchdb down' }] });
  render(<FeatureSelectionWidget />);
  searchDone(['a']);
  fireEvent.click(screen.getByRole('tab', { name: 'Structured Splits' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('couchdb down');
});
