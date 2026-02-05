/**
 * FeatureSelectionWidget - Tabbed widget with Vocabulary, Text Features, and JSON Features
 *
 * Provides three modes for adding terms to the Description field:
 * - Vocabulary: existing VocabTreeWidget (cascading selector)
 * - Text Features: ranked features from TaxaDecisionTreeClassifier
 * - JSON Features: ranked features from TaxaJsonClassifier
 *
 * Feature tabs show checkboxes ranked by decision tree importance.
 * Users select features and click "Add to Description" to append text
 * with trailing semicolons.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import VocabTreeWidget from './VocabTreeWidget';
import './FeatureSelectionWidget.css';

/**
 * Read the Django CSRF token from the cookie.
 */
function getCSRFToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
  return match ? match[1] : '';
}

/**
 * FeatureBrowserPanel - displays ranked features as checkboxes
 *
 * @param {Object} props
 * @param {Array} props.features - [{name, importance, display_text}, ...]
 * @param {Object} props.metadata - {n_classes, n_features, tree_depth, taxa_count}
 * @param {boolean} props.loading
 * @param {string|null} props.error
 * @param {Set} props.selectedFeatures - set of selected feature names
 * @param {Function} props.onToggleFeature - callback(featureName)
 * @param {Function} props.onAddToDescription - callback()
 * @param {string} props.panelId - unique id for aria
 * @param {string} props.tabId - associated tab id for aria
 * @param {string} props.label - panel label for screen readers
 */
const FeatureBrowserPanel = ({
  features,
  metadata,
  loading,
  error,
  selectedFeatures,
  onToggleFeature,
  onAddToDescription,
  panelId,
  tabId,
  label,
}) => {
  if (loading) {
    return (
      <div
        id={panelId}
        role="tabpanel"
        aria-labelledby={tabId}
        className="feature-panel"
      >
        <div role="status" aria-live="polite" className="feature-loading">
          Loading features...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        id={panelId}
        role="tabpanel"
        aria-labelledby={tabId}
        className="feature-panel"
      >
        <div role="alert" className="feature-error">
          {error}
        </div>
      </div>
    );
  }

  if (!features || features.length === 0) {
    return (
      <div
        id={panelId}
        role="tabpanel"
        aria-labelledby={tabId}
        className="feature-panel"
      >
        <div role="status" aria-live="polite" className="feature-empty">
          Run a search first to generate features.
        </div>
      </div>
    );
  }

  const maxImportance = features[0]?.importance || 1;
  const previewTexts = features
    .filter(f => selectedFeatures.has(f.name))
    .map(f => f.display_text + ';');

  return (
    <div
      id={panelId}
      role="tabpanel"
      aria-labelledby={tabId}
      className="feature-panel"
    >
      {metadata && (
        <div className="feature-metadata">
          {metadata.taxa_count} taxa, {metadata.n_features} features, depth {metadata.tree_depth}
        </div>
      )}

      <fieldset className="feature-fieldset">
        <legend className="sr-only">{label}</legend>
        <div className="feature-list">
          {features.map((feature, idx) => {
            const pct = ((feature.importance / maxImportance) * 100).toFixed(1);
            const isChecked = selectedFeatures.has(feature.name);
            return (
              <label key={feature.name} className="feature-item">
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => onToggleFeature(feature.name)}
                  aria-label={feature.display_text}
                />
                <span className="feature-name">{feature.display_text}</span>
                <span className="feature-bar-container">
                  <span
                    className="feature-bar"
                    style={{ width: `${pct}%` }}
                    aria-label={`Importance: ${pct}%`}
                    title={`${pct}% importance`}
                  />
                </span>
                <span className="feature-pct">{pct}%</span>
              </label>
            );
          })}
        </div>
      </fieldset>

      {previewTexts.length > 0 && (
        <div className="feature-preview">
          <span className="feature-preview-label">Preview:</span>
          <code className="feature-preview-text">{previewTexts.join(' ')}</code>
        </div>
      )}

      <div className="feature-actions">
        <button
          type="button"
          className="feature-add-btn"
          onClick={onAddToDescription}
          disabled={selectedFeatures.size === 0}
        >
          Add to Description
        </button>
      </div>
    </div>
  );
};

const TABS = [
  { id: 'vocabulary', label: 'Vocabulary' },
  { id: 'text-features', label: 'Text Features' },
  { id: 'json-features', label: 'JSON Features' },
];

/**
 * FeatureSelectionWidget Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {Function} props.onAddToDescription - Callback when "Add" is clicked
 * @param {React.RefObject} props.descriptionRef - Ref to description textarea
 * @param {string} props.version - Optional vocab tree version
 * @param {string} props.settingsSelector - CSS selector for settings container to observe
 */
const FeatureSelectionWidget = ({
  apiBaseUrl = '/api',
  onAddToDescription,
  descriptionRef,
  version,
  settingsSelector,
}) => {
  const [activeTab, setActiveTab] = useState('vocabulary');

  // Text features state
  const [textFeatures, setTextFeatures] = useState(null);
  const [textMetadata, setTextMetadata] = useState(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState(null);
  const [textSelected, setTextSelected] = useState(new Set());

  // JSON features state
  const [jsonFeatures, setJsonFeatures] = useState(null);
  const [jsonMetadata, setJsonMetadata] = useState(null);
  const [jsonLoading, setJsonLoading] = useState(false);
  const [jsonError, setJsonError] = useState(null);
  const [jsonSelected, setJsonSelected] = useState(new Set());

  // Taxa IDs from deeper search
  const [taxaIds, setTaxaIds] = useState(null);

  // Track which classifiers have been fetched for the current taxa set
  const fetchedForRef = useRef({ text: null, json: null });

  /**
   * Read classifier settings from DOM inputs (settings menu).
   * Falls back to sensible defaults if inputs are not found.
   */
  const readClassifierSettings = useCallback(() => {
    const topNEl = document.getElementById('featureTopN');
    const minDfEl = document.getElementById('featureMinDf');
    const maxDfEl = document.getElementById('featureMaxDf');
    const maxDepthEl = document.getElementById('featureMaxDepth');
    return {
      top_n: topNEl ? parseInt(topNEl.value, 10) || 30 : 30,
      min_df: minDfEl ? parseInt(minDfEl.value, 10) || 1 : 1,
      max_df: maxDfEl ? parseFloat(maxDfEl.value) || 1.0 : 1.0,
      max_depth: maxDepthEl ? parseInt(maxDepthEl.value, 10) || 10 : 10,
    };
  }, []);

  /**
   * Read taxa IDs from the deeper search cookie
   */
  const readTaxaCookie = useCallback(() => {
    const match = document.cookie.match(/(?:^|;\s*)skol_deeper_search=([^;]*)/);
    if (!match) return null;
    try {
      const data = JSON.parse(decodeURIComponent(match[1]));
      return data.taxa_ids || null;
    } catch {
      return null;
    }
  }, []);

  /**
   * Listen for deeper-search-complete events and cookie changes
   */
  useEffect(() => {
    const handleDeeperSearch = () => {
      const ids = readTaxaCookie();
      if (ids && ids.length > 0) {
        setTaxaIds(ids);
        // Reset fetched tracking so tabs re-fetch
        fetchedForRef.current = { text: null, json: null };
        // Clear previous results
        setTextFeatures(null);
        setTextMetadata(null);
        setTextError(null);
        setTextSelected(new Set());
        setJsonFeatures(null);
        setJsonMetadata(null);
        setJsonError(null);
        setJsonSelected(new Set());
      }
    };

    document.addEventListener('deeper-search-complete', handleDeeperSearch);

    // Also check cookie on mount
    const ids = readTaxaCookie();
    if (ids && ids.length > 0) {
      setTaxaIds(ids);
    }

    return () => {
      document.removeEventListener('deeper-search-complete', handleDeeperSearch);
    };
  }, [readTaxaCookie]);

  /**
   * Fetch features from a classifier endpoint
   */
  const fetchFeatures = useCallback(async (type, ids) => {
    const endpoint = type === 'text' ? 'classifier/text/' : 'classifier/json/';
    const setFeatures = type === 'text' ? setTextFeatures : setJsonFeatures;
    const setMeta = type === 'text' ? setTextMetadata : setJsonMetadata;
    const setLoading = type === 'text' ? setTextLoading : setJsonLoading;
    const setErr = type === 'text' ? setTextError : setJsonError;

    setLoading(true);
    setErr(null);

    try {
      const settings = readClassifierSettings();
      const response = await fetch(`${apiBaseUrl}/${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin',
        body: JSON.stringify({
          taxa_ids: ids,
          top_n: settings.top_n,
          max_depth: settings.max_depth,
          min_df: settings.min_df,
          max_df: settings.max_df,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setFeatures(data.features);
      setMeta(data.metadata);
      fetchedForRef.current[type] = ids;
    } catch (err) {
      console.error(`Failed to fetch ${type} features:`, err);
      setErr(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, readClassifierSettings]);

  /**
   * Fetch features when switching to a feature tab (lazy loading)
   */
  useEffect(() => {
    if (!taxaIds || taxaIds.length === 0) return;

    if (activeTab === 'text-features' && fetchedForRef.current.text !== taxaIds) {
      fetchFeatures('text', taxaIds);
    } else if (activeTab === 'json-features' && fetchedForRef.current.json !== taxaIds) {
      fetchFeatures('json', taxaIds);
    }
  }, [activeTab, taxaIds, fetchFeatures]);

  /**
   * Toggle a feature selection
   */
  const toggleFeature = useCallback((type, featureName) => {
    const setter = type === 'text' ? setTextSelected : setJsonSelected;
    setter(prev => {
      const next = new Set(prev);
      if (next.has(featureName)) {
        next.delete(featureName);
      } else {
        next.add(featureName);
      }
      return next;
    });
  }, []);

  /**
   * Add selected features to the description textarea
   */
  const handleAddFeatures = useCallback((type) => {
    const features = type === 'text' ? textFeatures : jsonFeatures;
    const selected = type === 'text' ? textSelected : jsonSelected;
    const setSel = type === 'text' ? setTextSelected : setJsonSelected;

    if (!features || selected.size === 0) return;

    const texts = features
      .filter(f => selected.has(f.name))
      .map(f => f.display_text + ';');
    const text = texts.join(' ') + ' ';

    if (onAddToDescription) {
      onAddToDescription(text);
    }

    if (descriptionRef?.current) {
      const textarea = descriptionRef.current;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const value = textarea.value;

      // If text before cursor doesn't end with ";" (and optional whitespace), prepend "; "
      const before = value.substring(0, start);
      const separator = before.length > 0 && !/;\s*$/.test(before) ? '; ' : '';
      const insert = separator + text;

      textarea.value = value.substring(0, start) + insert + value.substring(end);
      textarea.selectionStart = textarea.selectionEnd = start + insert.length;
      textarea.focus();

      const event = new Event('input', { bubbles: true });
      textarea.dispatchEvent(event);
    }

    setSel(new Set());
  }, [textFeatures, jsonFeatures, textSelected, jsonSelected, onAddToDescription, descriptionRef]);

  /**
   * Handle tab keyboard navigation (arrow keys)
   */
  const tabRefs = useRef({});
  const handleTabKeyDown = (e, tabId) => {
    const tabIds = TABS.map(t => t.id);
    const currentIndex = tabIds.indexOf(tabId);
    let newIndex = -1;

    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      newIndex = (currentIndex + 1) % tabIds.length;
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      newIndex = (currentIndex - 1 + tabIds.length) % tabIds.length;
    } else if (e.key === 'Home') {
      e.preventDefault();
      newIndex = 0;
    } else if (e.key === 'End') {
      e.preventDefault();
      newIndex = tabIds.length - 1;
    }

    if (newIndex >= 0) {
      setActiveTab(tabIds[newIndex]);
      tabRefs.current[tabIds[newIndex]]?.focus();
    }
  };

  return (
    <div className="feature-selection-widget">
      <div role="tablist" aria-label="Feature selection mode" className="feature-tabs">
        {TABS.map(tab => (
          <button
            type="button"
            key={tab.id}
            id={`tab-${tab.id}`}
            ref={el => { tabRefs.current[tab.id] = el; }}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            className={`feature-tab ${activeTab === tab.id ? 'feature-tab-active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            onKeyDown={(e) => handleTabKeyDown(e, tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Vocabulary panel */}
      <div
        id="panel-vocabulary"
        role="tabpanel"
        aria-labelledby="tab-vocabulary"
        hidden={activeTab !== 'vocabulary'}
      >
        {activeTab === 'vocabulary' && (
          <VocabTreeWidget
            apiBaseUrl={apiBaseUrl}
            onAddToDescription={onAddToDescription}
            descriptionRef={descriptionRef}
            version={version}
          />
        )}
      </div>

      {/* Text Features panel */}
      {activeTab === 'text-features' && (
        <FeatureBrowserPanel
          features={textFeatures}
          metadata={textMetadata}
          loading={textLoading}
          error={textError}
          selectedFeatures={textSelected}
          onToggleFeature={(name) => toggleFeature('text', name)}
          onAddToDescription={() => handleAddFeatures('text')}
          panelId="panel-text-features"
          tabId="tab-text-features"
          label="Text features ranked by importance"
        />
      )}

      {/* JSON Features panel */}
      {activeTab === 'json-features' && (
        <FeatureBrowserPanel
          features={jsonFeatures}
          metadata={jsonMetadata}
          loading={jsonLoading}
          error={jsonError}
          selectedFeatures={jsonSelected}
          onToggleFeature={(name) => toggleFeature('json', name)}
          onAddToDescription={() => handleAddFeatures('json')}
          panelId="panel-json-features"
          tabId="tab-json-features"
          label="JSON features ranked by importance"
        />
      )}
    </div>
  );
};

export default FeatureSelectionWidget;
