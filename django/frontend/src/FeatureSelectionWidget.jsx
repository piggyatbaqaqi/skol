/**
 * FeatureSelectionWidget - Tabbed widget of tools for the Description field
 *
 * - Vocabulary: existing VocabTreeWidget (cascading selector)
 * - Text Features: ranked features from TaxaDecisionTreeClassifier
 * - JSON Features: ranked features from TaxaJsonClassifier
 * - Text Splits, Structured Splits: how features divide the search
 *   results (docs/plans/result-splitting-features.md; ResultSplitsPanels)
 * - Metrics: MetricsWidget
 *
 * Feature tabs show checkboxes ranked by decision tree importance.
 * Users select features and click "Add to Description" to append text
 * with trailing semicolons.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import Select from 'react-select';
import VocabTreeWidget from './VocabTreeWidget';
import MetricsWidget from './MetricsWidget';
import { StructuredSplitsPanel, TextSplitsPanel } from './ResultSplitsPanels';
import { fitTabs, useTabMeasurements } from './tabOverflow';
import { insertIntoDescription } from './descriptionUtils';
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
 * @param {Object} props.metadata - {n_classes, n_features, tree_depth, treatments_count}
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
          {metadata.treatments_count} taxa, {metadata.n_features} features, depth {metadata.tree_depth}
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
  { id: 'structured-splits', label: 'Structured Splits' },
  { id: 'text-splits', label: 'Text Splits' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'text-features', label: 'Text Features' },
  { id: 'json-features', label: 'JSON Features' },
];

// The result-splitting tabs: tab id -> endpoint under apiBaseUrl.
const SPLITS_ENDPOINTS = {
  'text-splits': 'features/text/',
  'structured-splits': 'features/structured/',
};

/**
 * FeatureSelectionWidget Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {Function} props.onAddToDescription - Callback when "Add" is clicked
 * @param {React.RefObject} props.descriptionRef - Ref to description textarea
 * @param {string} props.version - Optional vocab tree version
 * @param {string} props.settingsSelector - CSS selector for settings container to observe
 * @param {number|string|null} props.collectionId - Active collection ID for metrics
 */
// Compact react-select styles for the "More" tab menu.
const MORE_SELECT_STYLES = {
  control: base => ({
    ...base, minHeight: 32, border: 'none', boxShadow: 'none',
    background: 'transparent', cursor: 'pointer',
  }),
  valueContainer: base => ({ ...base, padding: '0 4px' }),
  placeholder: base => ({ ...base, fontSize: 13, color: '#6c757d' }),
  indicatorSeparator: () => ({ display: 'none' }),
  dropdownIndicator: base => ({ ...base, padding: 4 }),
  menu: base => ({ ...base, width: 180, right: 0, zIndex: 20 }),
  menuList: base => ({ ...base, maxHeight: 240 }),
  option: base => ({ ...base, fontSize: 13 }),
};

const FeatureSelectionWidget = ({
  apiBaseUrl = '/api',
  onAddToDescription,
  descriptionRef,
  version,
  settingsSelector,
  collectionId,
}) => {
  const [activeTab, setActiveTab] = useState('vocabulary');

  // Tab overflow: tabs that do not fit go into the "More" menu.
  const {
    barRef, measureRef, available, widths, moreWidth,
  } = useTabMeasurements();
  const { visible: visibleTabIds, overflow: overflowTabIds } = fitTabs({
    ids: TABS.map(t => t.id), widths, available, moreWidth, active: activeTab,
  });
  const tabById = Object.fromEntries(TABS.map(t => [t.id, t]));

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

  // Result-splitting tabs, by tab id: response, loading, error
  const emptySplits = () => ({
    'text-splits': { data: null, loading: false, error: null },
    'structured-splits': { data: null, loading: false, error: null },
  });
  const [splits, setSplits] = useState(emptySplits);
  const splitsFetchedRef = useRef({});

  // Taxa IDs from search results
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
   * Listen for search-results-ready events from the basic search
   */
  useEffect(() => {
    const handleSearchResults = (event) => {
      const ids = event.detail?.treatment_ids;
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
        splitsFetchedRef.current = {};
        setSplits(emptySplits());
      }
    };

    document.addEventListener('search-results-ready', handleSearchResults);

    return () => {
      document.removeEventListener('search-results-ready', handleSearchResults);
    };
  }, []);

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
          treatment_ids: ids,
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
   * Fetch one result-splitting tab's scores for these results.
   */
  const fetchSplits = useCallback(async (tabId, ids) => {
    const update = (patch) => setSplits(prev => ({
      ...prev, [tabId]: { ...prev[tabId], ...patch },
    }));
    splitsFetchedRef.current[tabId] = ids;
    update({ loading: true, error: null });
    try {
      const response = await fetch(`${apiBaseUrl}/${SPLITS_ENDPOINTS[tabId]}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin',
        body: JSON.stringify({
          treatment_ids: ids,
          top_n: readClassifierSettings().top_n,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      update({ data, loading: false });
    } catch (err) {
      console.error(`Failed to fetch ${tabId}:`, err);
      update({ loading: false, error: err.message });
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
    } else if (activeTab in SPLITS_ENDPOINTS
               && splitsFetchedRef.current[activeTab] !== taxaIds) {
      fetchSplits(activeTab, taxaIds);
    }
  }, [activeTab, taxaIds, fetchFeatures, fetchSplits]);

  /**
   * Add text from a result-splitting tab, naming the tab as its source.
   */
  const addFromSplits = useCallback((text, source) => {
    if (onAddToDescription) {
      onAddToDescription(text, source);
    }
    if (descriptionRef?.current) {
      insertIntoDescription(descriptionRef.current, text);
    }
  }, [onAddToDescription, descriptionRef]);

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
    const source = type === 'text' ? 'text-features' : 'json-features';

    if (!features || selected.size === 0) return;

    const texts = features
      .filter(f => selected.has(f.name))
      .map(f => f.display_text + ';');
    const text = texts.join(' ') + ' ';

    if (onAddToDescription) {
      onAddToDescription(text, source);
    }

    if (descriptionRef?.current) {
      insertIntoDescription(descriptionRef.current, text);
    }

    setSel(new Set());
  }, [textFeatures, jsonFeatures, textSelected, jsonSelected, onAddToDescription, descriptionRef]);

  /**
   * Handle tab keyboard navigation (arrow keys)
   */
  const tabRefs = useRef({});
  // A tab chosen by keyboard may only appear in the bar on the next
  // render (it was in "More"), so focus it once it is there.
  const focusPendingRef = useRef(false);
  useEffect(() => {
    if (focusPendingRef.current) {
      focusPendingRef.current = false;
      tabRefs.current[activeTab]?.focus();
    }
  }, [activeTab]);
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
      focusPendingRef.current = true;
      setActiveTab(tabIds[newIndex]);
    }
  };

  return (
    <div className="feature-selection-widget">
      <div className="feature-tabs-bar" ref={barRef}>
      <div role="tablist" aria-label="Feature selection mode" className="feature-tabs">
        {visibleTabIds.map(id => tabById[id]).map(tab => (
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
      {/* Every label, measured off-screen, so the bar knows what fits;
          before "More", whose menu should come last in the page. */}
      <div className="feature-tabs-measure" ref={measureRef} aria-hidden="true">
        {TABS.map(tab => (
          <span key={tab.id} className="feature-tab feature-tab-measure">
            {tab.label}
          </span>
        ))}
        <span className="feature-tabs-more" data-more="true" />
      </div>
      {overflowTabIds.length > 0 && (
        <div className="feature-tabs-more">
          <Select
            aria-label="More tabs"
            placeholder="More"
            value={null}
            isSearchable={false}
            options={overflowTabIds.map(id => ({
              value: id, label: tabById[id].label,
            }))}
            onChange={option => option && setActiveTab(option.value)}
            styles={MORE_SELECT_STYLES}
            classNamePrefix="feature-more"
          />
        </div>
      )}
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
            onAddToDescription={(text) => onAddToDescription && onAddToDescription(text, 'vocabulary')}
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

      {/* Result-splitting panels: kept mounted, so ticks survive a tab
          switch, and hidden while another tab is active. */}
      <div
        id="panel-text-splits"
        role="tabpanel"
        aria-labelledby="tab-text-splits"
        className="feature-panel"
        hidden={activeTab !== 'text-splits'}
      >
        <TextSplitsPanel
          {...splits['text-splits']}
          onAddToDescription={(text) => addFromSplits(text, 'text-splits')}
        />
      </div>
      <div
        id="panel-structured-splits"
        role="tabpanel"
        aria-labelledby="tab-structured-splits"
        className="feature-panel"
        hidden={activeTab !== 'structured-splits'}
      >
        <StructuredSplitsPanel
          {...splits['structured-splits']}
          onAddToDescription={(text) => addFromSplits(text, 'structured-splits')}
        />
      </div>

      {/* Metrics panel */}
      {activeTab === 'metrics' && (
        <MetricsWidget
          apiBaseUrl={apiBaseUrl}
          collectionId={collectionId}
          onAddToDescription={(text) => onAddToDescription && onAddToDescription(text, 'metrics')}
          descriptionRef={descriptionRef}
        />
      )}
    </div>
  );
};

export default FeatureSelectionWidget;
