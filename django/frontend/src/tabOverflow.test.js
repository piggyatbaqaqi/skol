/**
 * Tests for fitTabs: which tabs fit in the widget's tab bar, and which
 * go into the "More" menu (user, 2026-11-23).
 *
 * test.failing marks tests that must fail until this is implemented, as
 * pytest's strict xfail does.
 */
import { fitTabs } from './tabOverflow';

const IDS = ['a', 'b', 'c', 'd'];
const HUNDREDS = [100, 100, 100, 100];

test('every tab shows when they all fit', () => {
  expect(fitTabs({ ids: IDS, widths: HUNDREDS, available: 400,
                   moreWidth: 60, active: 'a' }))
    .toEqual({ visible: IDS, overflow: [] });
});

test('every tab shows before the bar is measured', () => {
  expect(fitTabs({ ids: IDS, widths: [0, 0, 0, 0], available: 0,
                   moreWidth: 0, active: 'd' }))
    .toEqual({ visible: IDS, overflow: [] });
});

test.failing('tabs that do not fit go into More, leaving room for it', () => {
  // 330 - 60 for More leaves 270: two tabs of 100.
  expect(fitTabs({ ids: IDS, widths: HUNDREDS, available: 330,
                   moreWidth: 60, active: 'a' }))
    .toEqual({ visible: ['a', 'b'], overflow: ['c', 'd'] });
});

test.failing('the active tab stays in the bar, in its place', () => {
  expect(fitTabs({ ids: IDS, widths: HUNDREDS, available: 330,
                   moreWidth: 60, active: 'd' }))
    .toEqual({ visible: ['a', 'd'], overflow: ['b', 'c'] });
});

test.failing('the active tab shows even when it alone is too wide', () => {
  expect(fitTabs({ ids: ['a', 'b'], widths: [100, 500], available: 300,
                   moreWidth: 60, active: 'b' }))
    .toEqual({ visible: ['b'], overflow: ['a'] });
});

test.failing('tabs of different widths fill the bar in order', () => {
  // 270 of room: a (120) + b (90) = 210; c (80) would make 290.
  expect(fitTabs({ ids: IDS, widths: [120, 90, 80, 40], available: 330,
                   moreWidth: 60, active: 'a' }))
    .toEqual({ visible: ['a', 'b'], overflow: ['c', 'd'] });
});

test('tabs that exactly fill the bar need no More', () => {
  expect(fitTabs({ ids: IDS, widths: HUNDREDS, available: 400,
                   moreWidth: 60, active: 'c' }))
    .toEqual({ visible: IDS, overflow: [] });
});
