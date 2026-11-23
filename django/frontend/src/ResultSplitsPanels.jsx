/**
 * ResultSplitsPanels - the Structured Splits and Text Splits tools
 * (docs/plans/result-splitting-features.md, F3).
 *
 * Each panel shows how features divide the current search results, as
 * scored by POST /api/features/structured/ and /api/features/text/
 * (docs/api-reference.md, "Result splits"), and adds the ticked ones to
 * the description.
 *
 * Skeletons until the F3 tests are approved and implemented.
 */
import React from 'react';

/**
 * @param {Object} props
 * @param {Object|null} props.data - the /api/features/structured/ response
 * @param {boolean} props.loading
 * @param {string|null} props.error
 * @param {Function} props.onAddToDescription - callback(text)
 */
export const StructuredSplitsPanel = () => null;

/**
 * @param {Object} props
 * @param {Object|null} props.data - the /api/features/text/ response
 * @param {boolean} props.loading
 * @param {string|null} props.error
 * @param {Function} props.onAddToDescription - callback(text)
 */
export const TextSplitsPanel = () => null;
