/**
 * MetricsWidget - Measurement entry and statistics tool
 *
 * Records specimen measurements (e.g., spore dimensions) and computes
 * quartile-based statistics in standard mycological notation:
 *   spores (min-) Q1 - Q3 (-max) × (min-) Q1 - Q3 (-max) µm, Q: (min-)Q1-Q3(max)
 *
 * Raw measurements are stored per-collection via the measurements API.
 * The formatted summary can be inserted into the description textarea.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { insertIntoDescription } from './descriptionUtils';
import './MetricsWidget.css';

/**
 * Read the Django CSRF token from the cookie.
 */
function getCSRFToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
  return match ? match[1] : '';
}

/**
 * Compute the p-th percentile of a sorted array using linear interpolation.
 */
function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/**
 * Format a number for display: up to 1 decimal place, no trailing zeros.
 */
function fmt(n) {
  return parseFloat(n.toFixed(1)).toString();
}

/**
 * Compute statistics and formatted text from raw measurements.
 */
function computeStats(measurements, is2d, reportQ, featureName, unit) {
  const n = measurements.length;
  if (n === 0) return { text: '', n: 0 };

  const lengths = measurements.map(m => m.length).sort((a, b) => a - b);

  let text = `${featureName} (${fmt(lengths[0])}-) ${fmt(percentile(lengths, 25))} - ${fmt(percentile(lengths, 75))} (-${fmt(lengths[lengths.length - 1])})`;

  if (is2d) {
    const widths = measurements
      .filter(m => m.width != null)
      .map(m => m.width)
      .sort((a, b) => a - b);

    if (widths.length > 0) {
      text += ` \u00d7 (${fmt(widths[0])}-) ${fmt(percentile(widths, 25))} - ${fmt(percentile(widths, 75))} (-${fmt(widths[widths.length - 1])}) ${unit}`;

      if (reportQ) {
        const qs = measurements
          .filter(m => m.width != null && m.width > 0)
          .map(m => m.length / m.width)
          .sort((a, b) => a - b);

        if (qs.length > 0) {
          text += `, Q: (${fmt(qs[0])}-)${fmt(percentile(qs, 25))}-${fmt(percentile(qs, 75))}(${fmt(qs[qs.length - 1])})`;
        }
      }
    } else {
      text += ` ${unit}`;
    }
  } else {
    text += ` ${unit}`;
  }

  return { text, n };
}

/**
 * MetricsWidget Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Base URL for API calls
 * @param {number|string|null} props.collectionId - Active collection ID
 * @param {Function} props.onAddToDescription - Callback when "Add" is clicked
 * @param {React.RefObject} props.descriptionRef - Ref to description textarea
 */
const MetricsWidget = ({
  apiBaseUrl = '/api',
  collectionId,
  onAddToDescription,
  descriptionRef,
}) => {
  // All measurement sets for this collection
  const [measurementSets, setMeasurementSets] = useState([]);
  // Currently active measurement set
  const [activeSet, setActiveSet] = useState(null);
  // Local working copy of measurements
  const [measurements, setMeasurements] = useState([]);
  // Feature name input
  const [featureName, setFeatureName] = useState('spores');
  // Settings
  const [is2d, setIs2d] = useState(true);
  const [reportQ, setReportQ] = useState(true);
  const [unit, setUnit] = useState('\u00b5m');
  // Available units from admin configuration
  const [availableUnits, setAvailableUnits] = useState(['\u00b5m']);
  // Sample entry
  const [inputA, setInputA] = useState('');
  const [inputB, setInputB] = useState('');
  // Selected sample index for edit/remove
  const [selectedSample, setSelectedSample] = useState(null);
  // Loading/error state
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  // Track if we've loaded for this collection
  const loadedForRef = useRef(null);
  // Debounce timer for auto-save
  const saveTimerRef = useRef(null);
  // Ref for first input field
  const inputARef = useRef(null);

  /**
   * Fetch available measurement units from the server
   */
  useEffect(() => {
    (async () => {
      try {
        const response = await fetch(
          `${apiBaseUrl}/measurement-units/`,
          { credentials: 'same-origin' }
        );
        if (response.ok) {
          const data = await response.json();
          if (data.units && data.units.length > 0) {
            setAvailableUnits(data.units);
          }
        }
      } catch (err) {
        console.error('Failed to fetch measurement units:', err);
      }
    })();
  }, [apiBaseUrl]);

  /**
   * Fetch all measurement sets for the active collection
   */
  const fetchMeasurementSets = useCallback(async (colId) => {
    if (!colId) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${apiBaseUrl}/collections/${colId}/measurements/`,
        { credentials: 'same-origin' }
      );
      if (!response.ok) {
        if (response.status === 403) {
          setMeasurementSets([]);
          setLoading(false);
          return;
        }
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setMeasurementSets(data.measurement_sets || []);

      // Load the first set if available
      if (data.measurement_sets && data.measurement_sets.length > 0) {
        const first = data.measurement_sets[0];
        setActiveSet(first);
        setFeatureName(first.feature);
        setIs2d(first.is_2d);
        setReportQ(first.report_q);
        setUnit(first.unit || '\u00b5m');
        setMeasurements(first.measurements || []);
      } else {
        setActiveSet(null);
        setFeatureName('spores');
        setIs2d(true);
        setReportQ(true);
        setUnit('\u00b5m');
        setMeasurements([]);
      }
      loadedForRef.current = colId;
    } catch (err) {
      console.error('Failed to fetch measurement sets:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  /**
   * Load measurement sets when collection changes
   */
  useEffect(() => {
    if (collectionId && loadedForRef.current !== collectionId) {
      fetchMeasurementSets(collectionId);
    }
  }, [collectionId, fetchMeasurementSets]);

  /**
   * Save current measurement set to the server (create or update)
   */
  const saveToServer = useCallback(async (ms, feature, d2, rq, u) => {
    if (!collectionId) return;
    setSaving(true);
    try {
      if (ms && ms.id) {
        // Update existing
        const response = await fetch(
          `${apiBaseUrl}/collections/${collectionId}/measurements/${ms.id}/`,
          {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
              'X-CSRFToken': getCSRFToken(),
            },
            credentials: 'same-origin',
            body: JSON.stringify({
              feature: feature,
              is_2d: d2,
              report_q: rq,
              unit: u,
              measurements: ms.measurements,
            }),
          }
        );
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.error || `HTTP ${response.status}`);
        }
        const updated = await response.json();
        setActiveSet(updated);
        // Update in the list
        setMeasurementSets(prev =>
          prev.map(s => (s.id === updated.id ? updated : s))
        );
      } else {
        // Create new
        const response = await fetch(
          `${apiBaseUrl}/collections/${collectionId}/measurements/`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-CSRFToken': getCSRFToken(),
            },
            credentials: 'same-origin',
            body: JSON.stringify({
              feature: feature,
              is_2d: d2,
              report_q: rq,
              unit: u,
              measurements: ms ? ms.measurements : [],
            }),
          }
        );
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.error || `HTTP ${response.status}`);
        }
        const created = await response.json();
        setActiveSet(created);
        setMeasurementSets(prev => [...prev, created]);
      }
    } catch (err) {
      console.error('Failed to save measurement set:', err);
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }, [apiBaseUrl, collectionId]);

  /**
   * Auto-save with debounce when measurements, settings, or feature name change
   */
  const scheduleSave = useCallback((updatedMeasurements, feature, d2, rq, u) => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = setTimeout(() => {
      const setToSave = activeSet
        ? { ...activeSet, measurements: updatedMeasurements }
        : { measurements: updatedMeasurements };
      saveToServer(setToSave, feature, d2, rq, u);
    }, 500);
  }, [activeSet, saveToServer]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, []);

  /**
   * Add a new sample from the input fields
   */
  const handleAddSample = useCallback(() => {
    const a = parseFloat(inputA);
    const b = is2d ? parseFloat(inputB) : null;

    if (isNaN(a) || a <= 0) return;
    if (is2d && (isNaN(b) || b <= 0)) return;

    // Larger value is length
    let length, width;
    if (is2d && b != null) {
      length = Math.max(a, b);
      width = Math.min(a, b);
    } else {
      length = a;
      width = null;
    }

    const newMeasurement = { length };
    if (width != null) newMeasurement.width = width;

    const updated = [...measurements, newMeasurement];
    setMeasurements(updated);
    setInputA('');
    setInputB('');
    setSelectedSample(null);
    scheduleSave(updated, featureName, is2d, reportQ, unit);

    // Focus back to first input
    if (inputARef.current) inputARef.current.focus();
  }, [inputA, inputB, is2d, measurements, featureName, reportQ, unit, scheduleSave]);

  /**
   * Handle Enter key in input fields
   */
  const handleInputKeyDown = useCallback((e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddSample();
    }
  }, [handleAddSample]);

  /**
   * Update an existing sample
   */
  const handleUpdateSample = useCallback(() => {
    if (selectedSample === null) return;
    const a = parseFloat(inputA);
    const b = is2d ? parseFloat(inputB) : null;

    if (isNaN(a) || a <= 0) return;
    if (is2d && (isNaN(b) || b <= 0)) return;

    let length, width;
    if (is2d && b != null) {
      length = Math.max(a, b);
      width = Math.min(a, b);
    } else {
      length = a;
      width = null;
    }

    const newMeasurement = { length };
    if (width != null) newMeasurement.width = width;

    const updated = [...measurements];
    updated[selectedSample] = newMeasurement;
    setMeasurements(updated);
    setSelectedSample(null);
    setInputA('');
    setInputB('');
    scheduleSave(updated, featureName, is2d, reportQ, unit);
  }, [selectedSample, inputA, inputB, is2d, measurements, featureName, reportQ, unit, scheduleSave]);

  /**
   * Remove the selected sample
   */
  const handleRemoveSample = useCallback(() => {
    if (selectedSample === null) return;
    const updated = measurements.filter((_, i) => i !== selectedSample);
    setMeasurements(updated);
    setSelectedSample(null);
    setInputA('');
    setInputB('');
    scheduleSave(updated, featureName, is2d, reportQ, unit);
  }, [selectedSample, measurements, featureName, is2d, reportQ, unit, scheduleSave]);

  /**
   * Select a sample from the dropdown to edit
   */
  const handleSelectSample = useCallback((e) => {
    const idx = parseInt(e.target.value, 10);
    if (isNaN(idx) || idx < 0) {
      setSelectedSample(null);
      setInputA('');
      setInputB('');
      return;
    }
    const m = measurements[idx];
    setSelectedSample(idx);
    setInputA(m.length.toString());
    setInputB(m.width != null ? m.width.toString() : '');
  }, [measurements]);

  /**
   * Switch to a different measurement set
   */
  const handleSwitchSet = useCallback((e) => {
    const id = parseInt(e.target.value, 10);
    if (isNaN(id)) {
      // "New" was selected
      setActiveSet(null);
      setFeatureName('spores');
      setIs2d(true);
      setReportQ(true);
      setUnit('\u00b5m');
      setMeasurements([]);
      setSelectedSample(null);
      setInputA('');
      setInputB('');
      return;
    }
    const set = measurementSets.find(s => s.id === id);
    if (set) {
      setActiveSet(set);
      setFeatureName(set.feature);
      setIs2d(set.is_2d);
      setReportQ(set.report_q);
      setUnit(set.unit || '\u00b5m');
      setMeasurements(set.measurements || []);
      setSelectedSample(null);
      setInputA('');
      setInputB('');
    }
  }, [measurementSets]);

  /**
   * Delete the active measurement set
   */
  const handleDeleteSet = useCallback(async () => {
    if (!activeSet || !activeSet.id || !collectionId) return;
    const n = measurements.length;
    const msg = `Delete "${featureName}" set with ${n} sample${n !== 1 ? 's' : ''}? This cannot be undone.`;
    if (!window.confirm(msg)) return;
    try {
      const response = await fetch(
        `${apiBaseUrl}/collections/${collectionId}/measurements/${activeSet.id}/`,
        {
          method: 'DELETE',
          headers: { 'X-CSRFToken': getCSRFToken() },
          credentials: 'same-origin',
        }
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      setMeasurementSets(prev => prev.filter(s => s.id !== activeSet.id));
      setActiveSet(null);
      setFeatureName('spores');
      setMeasurements([]);
      setSelectedSample(null);
    } catch (err) {
      setError(err.message);
    }
  }, [activeSet, apiBaseUrl, collectionId, measurements, featureName]);

  /**
   * Handle feature name change with debounced save
   */
  const handleFeatureNameChange = useCallback((e) => {
    const name = e.target.value;
    setFeatureName(name);
    scheduleSave(measurements, name, is2d, reportQ, unit);
  }, [measurements, is2d, reportQ, unit, scheduleSave]);

  /**
   * Handle dimension toggle
   */
  const handleDimensionChange = useCallback((newIs2d) => {
    setIs2d(newIs2d);
    scheduleSave(measurements, featureName, newIs2d, reportQ, unit);
  }, [measurements, featureName, reportQ, unit, scheduleSave]);

  /**
   * Handle report Q toggle
   */
  const handleReportQChange = useCallback((e) => {
    const newReportQ = e.target.checked;
    setReportQ(newReportQ);
    scheduleSave(measurements, featureName, is2d, newReportQ, unit);
  }, [measurements, featureName, is2d, unit, scheduleSave]);

  /**
   * Handle unit change
   */
  const handleUnitChange = useCallback((e) => {
    const newUnit = e.target.value;
    setUnit(newUnit);
    scheduleSave(measurements, featureName, is2d, reportQ, newUnit);
  }, [measurements, featureName, is2d, reportQ, scheduleSave]);

  /**
   * Add formatted statistics to the description textarea
   */
  const handleAddToDescription = useCallback(() => {
    const { text } = computeStats(measurements, is2d, reportQ, featureName, unit);
    if (!text) return;

    const insertText = text + '; ';

    if (onAddToDescription) {
      onAddToDescription(insertText);
    }

    if (descriptionRef?.current) {
      insertIntoDescription(descriptionRef.current, insertText);
    }
  }, [measurements, is2d, reportQ, featureName, unit, onAddToDescription, descriptionRef]);

  // Compute current statistics
  const stats = computeStats(measurements, is2d, reportQ, featureName, unit);

  // No collection selected
  if (!collectionId) {
    return (
      <div className="metrics-widget">
        <div className="metrics-empty">
          Select a collection to record measurements.
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="metrics-widget">
        <div className="metrics-loading">Loading measurements...</div>
      </div>
    );
  }

  return (
    <div className="metrics-widget">
      {error && (
        <div className="metrics-error" role="alert">
          {error}
          <button
            type="button"
            className="metrics-error-dismiss"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
          >
            &times;
          </button>
        </div>
      )}

      {/* Feature name + set selector */}
      <div className="metrics-header">
        <div className="metrics-feature-row">
          <label className="metrics-label" htmlFor="metrics-feature">
            Feature:
          </label>
          <input
            id="metrics-feature"
            type="text"
            className="metrics-feature-input"
            value={featureName}
            onChange={handleFeatureNameChange}
            placeholder="spores"
          />
          {measurementSets.length > 0 && (
            <select
              className="metrics-set-select"
              value={activeSet ? activeSet.id : ''}
              onChange={handleSwitchSet}
              aria-label="Select measurement set"
            >
              {measurementSets.map(s => (
                <option key={s.id} value={s.id}>
                  {s.feature} ({(s.measurements || []).length})
                </option>
              ))}
              <option value="">+ New</option>
            </select>
          )}
        </div>
      </div>

      {/* Statistics display */}
      {stats.n > 0 && (
        <div className="metrics-stats-box">
          <code className="metrics-stats-text">{stats.text}</code>
        </div>
      )}

      {/* Sample count */}
      <div className="metrics-n-row">
        <span className={`metrics-n ${stats.n < 30 ? 'metrics-n-warn' : 'metrics-n-ok'}`}>
          n = {stats.n}
        </span>
        {stats.n > 0 && stats.n < 30 && (
          <span className="metrics-n-hint">(target: 30)</span>
        )}
        {saving && <span className="metrics-saving">Saving...</span>}
      </div>

      {/* Dimension toggle */}
      <div className="metrics-dimension-row">
        <span className="metrics-label">Dimensions:</span>
        <label className="metrics-radio">
          <input
            type="radio"
            name="metrics-dim"
            checked={is2d}
            onChange={() => handleDimensionChange(true)}
          />
          2D
        </label>
        <label className="metrics-radio">
          <input
            type="radio"
            name="metrics-dim"
            checked={!is2d}
            onChange={() => handleDimensionChange(false)}
          />
          1D
        </label>
      </div>

      {/* Measurement entry */}
      <div className="metrics-entry-row">
        <input
          ref={inputARef}
          type="number"
          step="0.1"
          min="0"
          className="metrics-num-input"
          value={inputA}
          onChange={(e) => setInputA(e.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder="L"
          aria-label="Length or first dimension"
        />
        {is2d && (
          <>
            <span className="metrics-times">&times;</span>
            <input
              type="number"
              step="0.1"
              min="0"
              className="metrics-num-input"
              value={inputB}
              onChange={(e) => setInputB(e.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder="W"
              aria-label="Width or second dimension"
            />
          </>
        )}
        <select
          className="metrics-unit-select"
          value={unit}
          onChange={handleUnitChange}
          aria-label="Unit of measurement"
        >
          {availableUnits.map(u => (
            <option key={u} value={u}>{u}</option>
          ))}
        </select>
        {selectedSample === null ? (
          <button
            type="button"
            className="metrics-add-sample-btn"
            onClick={handleAddSample}
            disabled={!inputA || (is2d && !inputB)}
            title="Add sample"
            aria-label="Add sample"
          >
            +
          </button>
        ) : (
          <button
            type="button"
            className="metrics-update-sample-btn"
            onClick={handleUpdateSample}
            disabled={!inputA || (is2d && !inputB)}
            title="Update sample"
            aria-label="Update sample"
          >
            &#10003;
          </button>
        )}
      </div>

      {/* Report Q checkbox */}
      {is2d && (
        <label className="metrics-report-q">
          <input
            type="checkbox"
            checked={reportQ}
            onChange={handleReportQChange}
          />
          Report Q values
        </label>
      )}

      {/* Sample list */}
      {measurements.length > 0 && (
        <div className="metrics-samples-row">
          <label className="metrics-label" htmlFor="metrics-sample-select">
            Samples:
          </label>
          <select
            id="metrics-sample-select"
            className="metrics-sample-select"
            value={selectedSample !== null ? selectedSample : ''}
            onChange={handleSelectSample}
          >
            <option value="">Select sample...</option>
            {measurements.map((m, i) => (
              <option key={i} value={i}>
                #{i + 1}: {fmt(m.length)}{m.width != null ? ` \u00d7 ${fmt(m.width)}` : ''}
              </option>
            ))}
          </select>
          {selectedSample !== null && (
            <div className="metrics-sample-actions">
              <button
                type="button"
                className="metrics-btn-small metrics-btn-edit"
                onClick={() => {
                  // Already loaded into inputs by handleSelectSample
                }}
                disabled
              >
                Edit
              </button>
              <button
                type="button"
                className="metrics-btn-small metrics-btn-remove"
                onClick={handleRemoveSample}
              >
                Remove
              </button>
            </div>
          )}
        </div>
      )}

      {/* Delete set button */}
      {activeSet && activeSet.id && (
        <div className="metrics-delete-row">
          <button
            type="button"
            className="metrics-btn-small metrics-btn-remove"
            onClick={handleDeleteSet}
          >
            Delete &ldquo;{featureName}&rdquo; set
          </button>
        </div>
      )}

      {/* Add to Description */}
      <div className="metrics-actions">
        <button
          type="button"
          className="metrics-add-btn"
          onClick={handleAddToDescription}
          disabled={stats.n === 0}
        >
          Add to Description
        </button>
      </div>
    </div>
  );
};

export default MetricsWidget;
