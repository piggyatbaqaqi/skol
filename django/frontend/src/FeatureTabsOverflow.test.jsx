/**
 * Tests for the widget's tab overflow: tabs that do not fit the 400px
 * bar go into a "More" menu (react-select), and the active tab always
 * stays in the bar (user, 2026-11-23).
 *
 * Measuring needs a browser, so useTabMeasurements is replaced here by
 * fixed widths: six tabs of 100px, a "More" control of 60px.
 *
 * Approved by the user 2026-11-23; they ran as test.failing (Jest's
 * strict xfail) before the overflow (d65cfec) and the tab reorder.
 */
import { fireEvent, render, screen } from '@testing-library/react';
import selectEvent from 'react-select-event';
import FeatureSelectionWidget from './FeatureSelectionWidget';

jest.mock('./VocabTreeWidget', () => () => <div>vocabulary panel</div>);
jest.mock('./MetricsWidget', () => () => <div>metrics panel</div>);

const mockLayout = { available: 0 };
jest.mock('./tabOverflow', () => {
  const actual = jest.requireActual('./tabOverflow');
  return {
    ...actual,
    useTabMeasurements: () => ({
      barRef: { current: null },
      measureRef: { current: null },
      available: mockLayout.available,
      widths: [100, 100, 100, 100, 100, 100],
      moreWidth: 60,
    }),
  };
});

function tabNames() {
  return screen.getAllByRole('tab').map(t => t.textContent);
}

function moreMenu() {
  return screen.getByRole('combobox', { name: 'More tabs' });
}

async function moreOptions() {
  selectEvent.openMenu(moreMenu());
  return screen.getAllByRole('option').map(o => o.textContent);
}

test('every tab shows when they all fit', () => {
  mockLayout.available = 1000;
  render(<FeatureSelectionWidget />);
  expect(tabNames()).toEqual([
    'Vocabulary', 'Structured Splits', 'Text Splits', 'Metrics',
    'Text Features', 'JSON Features',
  ]);
  expect(screen.queryByRole('combobox', { name: 'More tabs' })).toBeNull();
});

test('tabs that do not fit go into More', async () => {
  // 330 - 60 for More leaves room for two tabs.
  mockLayout.available = 330;
  render(<FeatureSelectionWidget />);
  expect(tabNames()).toEqual(['Vocabulary', 'Structured Splits']);
  expect(await moreOptions()).toEqual([
    'Text Splits', 'Metrics', 'Text Features', 'JSON Features',
  ]);
});

test('choosing from More opens that tab and keeps it in the bar', async () => {
  mockLayout.available = 330;
  render(<FeatureSelectionWidget />);
  await selectEvent.select(moreMenu(), 'Metrics');
  expect(tabNames()).toEqual(['Vocabulary', 'Metrics']);
  expect(screen.getByRole('tab', { name: 'Metrics' }))
    .toHaveAttribute('aria-selected', 'true');
  expect(screen.getByText('metrics panel')).toBeInTheDocument();
  expect(await moreOptions()).toEqual([
    'Structured Splits', 'Text Splits', 'Text Features', 'JSON Features',
  ]);
});

test('arrow keys reach the tabs in More', () => {
  mockLayout.available = 330;
  render(<FeatureSelectionWidget />);
  const structuredTab = screen.getByRole('tab', { name: 'Structured Splits' });
  fireEvent.click(structuredTab);
  fireEvent.keyDown(structuredTab, { key: 'ArrowRight' });
  const textTab = screen.getByRole('tab', { name: 'Text Splits' });
  expect(textTab).toHaveAttribute('aria-selected', 'true');
  expect(textTab).toHaveFocus();
  expect(tabNames()).toEqual(['Vocabulary', 'Text Splits']);
});

test('End reaches the last tab, wherever it is', () => {
  mockLayout.available = 330;
  render(<FeatureSelectionWidget />);
  fireEvent.keyDown(screen.getByRole('tab', { name: 'Vocabulary' }),
                    { key: 'End' });
  expect(screen.getByRole('tab', { name: 'JSON Features' }))
    .toHaveAttribute('aria-selected', 'true');
  expect(tabNames()).toEqual(['Vocabulary', 'JSON Features']);
});
