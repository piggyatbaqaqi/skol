/**
 * Tab overflow: which tabs fit in the bar, and which go into "More".
 *
 * The feature-selection widget is 400px wide, too narrow for all its
 * tab labels (user, 2026-11-23).  Tabs that do not fit go into a "More"
 * menu; the active tab always stays in the bar.
 *
 * Skeletons until the tests are approved and implemented.
 */
import { useRef } from 'react';

/**
 * Split tabs between the bar and the "More" menu.
 *
 * @param {Object} args
 * @param {string[]} args.ids - tab ids, in bar order
 * @param {number[]} args.widths - each tab's width, in px
 * @param {number} args.available - the bar's width; 0 if not measured
 * @param {number} args.moreWidth - the "More" control's width
 * @param {string} args.active - the active tab's id
 * @returns {{visible: string[], overflow: string[]}} both in bar order
 */
export function fitTabs({ ids }) {
  return { visible: [...ids], overflow: [] };
}

/**
 * Measure the tab bar, every tab label and the "More" control.
 *
 * @returns {{barRef, measureRef, available: number, widths: number[],
 *            moreWidth: number}} ``available`` is 0 until measured
 *            (and always where layout is unavailable, as in jsdom).
 */
export function useTabMeasurements() {
  const barRef = useRef(null);
  const measureRef = useRef(null);
  return { barRef, measureRef, available: 0, widths: [], moreWidth: 0 };
}
