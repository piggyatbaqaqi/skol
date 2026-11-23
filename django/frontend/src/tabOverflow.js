/**
 * Tab overflow: which tabs fit in the bar, and which go into "More".
 *
 * The feature-selection widget is 400px wide, too narrow for all its
 * tab labels (user, 2026-11-23).  Tabs that do not fit go into a "More"
 * menu; the active tab always stays in the bar.
 */
import { useLayoutEffect, useRef, useState } from 'react';

/**
 * Split tabs between the bar and the "More" menu.
 *
 * Every tab shows when they all fit, or when the bar is not measured
 * (``available`` 0).  Otherwise the active tab is placed first, then the
 * others fill the room left beside "More", in bar order, stopping at the
 * first that does not fit.  Both lists keep bar order.
 *
 * @param {Object} args
 * @param {string[]} args.ids - tab ids, in bar order
 * @param {number[]} args.widths - each tab's width, in px
 * @param {number} args.available - the bar's width; 0 if not measured
 * @param {number} args.moreWidth - the "More" control's width
 * @param {string} args.active - the active tab's id
 * @returns {{visible: string[], overflow: string[]}} both in bar order
 */
export function fitTabs({ ids, widths, available, moreWidth, active }) {
  const width = i => widths[i] || 0;
  const total = ids.reduce((sum, _, i) => sum + width(i), 0);
  if (!(available > 0) || total <= available) {
    return { visible: [...ids], overflow: [] };
  }
  const room = available - moreWidth;
  const chosen = new Set();
  let used = 0;
  const activeIndex = ids.indexOf(active);
  if (activeIndex >= 0) {
    chosen.add(activeIndex);
    used += width(activeIndex);
  }
  for (let i = 0; i < ids.length; i += 1) {
    if (chosen.has(i)) continue;
    if (used + width(i) > room) break;
    chosen.add(i);
    used += width(i);
  }
  return {
    visible: ids.filter((_, i) => chosen.has(i)),
    overflow: ids.filter((_, i) => !chosen.has(i)),
  };
}

/**
 * Measure the tab bar, every tab label and the "More" control.
 *
 * ``barRef`` goes on the element whose width is the room for tabs and
 * "More"; ``measureRef`` on a hidden row holding every tab label in bar
 * order, then one element marked ``data-more`` sized like "More".  The
 * row is measured, not the bar's tabs, so the widths do not depend on
 * which tabs are showing and the bar cannot oscillate.
 *
 * @returns {{barRef, measureRef, available: number, widths: number[],
 *            moreWidth: number}} ``available`` is 0 until measured
 *            (and always where layout is unavailable, as in jsdom).
 */
export function useTabMeasurements() {
  const barRef = useRef(null);
  const measureRef = useRef(null);
  const [size, setSize] = useState({ available: 0, widths: [], moreWidth: 0 });

  useLayoutEffect(() => {
    const bar = barRef.current;
    const row = measureRef.current;
    if (!bar || !row || typeof ResizeObserver === 'undefined') {
      return undefined;
    }
    const measure = () => {
      const items = Array.from(row.children);
      const more = items.find(el => el.dataset.more);
      const next = {
        available: bar.clientWidth,
        widths: items.filter(el => !el.dataset.more).map(el => el.offsetWidth),
        moreWidth: more ? more.offsetWidth : 0,
      };
      setSize(prev => (
        prev.available === next.available
        && prev.moreWidth === next.moreWidth
        && prev.widths.join() === next.widths.join()
      ) ? prev : next);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(bar);
    observer.observe(row);
    return () => observer.disconnect();
  }, []);

  return { barRef, measureRef, ...size };
}
