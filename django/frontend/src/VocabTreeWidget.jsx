/**
 * VocabTreeWidget - Cascading vocabulary tree selector
 *
 * A widget that displays cascading dropdown menus for navigating the
 * vocabulary tree. Each selection populates the next level until
 * reaching a leaf node. Selected terms can be added to a description
 * text field.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import Select from 'react-select';
import { insertIntoDescription } from './descriptionUtils';
import { getCSRFToken } from './csrf';
import './VocabTreeWidget.css';

/**
 * Custom styles for react-select to match the widget theme
 */
const selectStyles = {
  control: (base, state) => ({
    ...base,
    minHeight: '36px',
    borderColor: state.isFocused ? '#4a9eff' : '#ccc',
    boxShadow: state.isFocused ? '0 0 0 1px #4a9eff' : 'none',
    '&:hover': {
      borderColor: '#4a9eff'
    }
  }),
  option: (base, state) => ({
    ...base,
    backgroundColor: state.isSelected
      ? '#4a9eff'
      : state.isFocused
        ? '#e6f3ff'
        : 'white',
    color: state.isSelected ? 'white' : '#333',
    cursor: 'pointer',
    '&:active': {
      backgroundColor: '#4a9eff'
    }
  }),
  menu: (base) => ({
    ...base,
    zIndex: 100
  }),
  placeholder: (base) => ({
    ...base,
    color: '#999'
  })
};

// While a build the widget started is running, poll GET build/ this often,
// for at most this long (a production build takes about a minute).
const BUILD_POLL_MS = 2000;
const BUILD_WAIT_MS = 3 * 60 * 1000;
// GET build/ answers 'building' while the build holds its lock.  Before
// the build process has taken the lock it answers 'not_found'; after
// this long, or once it has answered 'building', 'not_found' means the
// build ended without a tree: it failed.
const BUILD_LOCK_GRACE_MS = 15 * 1000;
const BUILD_LOG = '/var/log/skol/build-vocab-tree-api.log';

const countFormat = new Intl.NumberFormat(
  (typeof navigator !== 'undefined' && navigator.language) || 'en-US');

const capitalize = (term) => term.charAt(0).toUpperCase() + term.slice(1);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * VocabTreeWidget Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {Function} props.onAddToDescription - Callback when "Add" is clicked
 * @param {React.RefObject} props.descriptionRef - Ref to description textarea
 * @param {string} props.version - Optional vocab tree version
 */
const VocabTreeWidget = ({
  apiBaseUrl = '/api',
  onAddToDescription,
  descriptionRef,
  version
}) => {
  // State for each level's data and selections
  const [levels, setLevels] = useState([]);
  const [selections, setSelections] = useState([]);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);
  const [building, setBuilding] = useState(false);
  // The build outlasted BUILD_WAIT_MS: a notice, not an error.
  const [stillBuilding, setStillBuilding] = useState(false);
  // The build ended without a tree.
  const [buildFailed, setBuildFailed] = useState(false);
  // Bumped by Retry to load the top level again.
  const [attempt, setAttempt] = useState(0);

  // Store the original top-level data for resets
  const topLevelDataRef = useRef(null);

  /**
   * Fetch children at a given path from the vocab tree API
   */
  const fetchChildren = useCallback(async (path = '') => {
    const params = new URLSearchParams();
    if (path) params.append('path', path);
    if (version) params.append('version', version);

    const response = await fetch(`${apiBaseUrl}/vocab-tree/children/?${params}`);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to fetch vocabulary tree');
    }
    return response.json();
  }, [apiBaseUrl, version]);

  /**
   * Trigger building the vocabulary tree if it doesn't exist.
   *
   * Resolves to 'ready' once the tree exists, 'timeout' when a build the
   * server started is still running after BUILD_WAIT_MS, 'failed' when
   * it ended without a tree, or 'error'.
   * POST answers 202 'building' when it starts a build; the tree is then
   * polled with GET until it reports 'exists'.
   */
  const triggerBuild = useCallback(async () => {
    setBuilding(true);
    setError(null);
    try {
      // Signed-in users' POSTs need the CSRF token, or DRF answers 403.
      const response = await fetch(`${apiBaseUrl}/vocab-tree/build/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin',
        body: JSON.stringify({})
      });
      const data = await response.json();

      if (data.status === 'complete' || data.status === 'exists') {
        return 'ready';
      }
      if (data.status !== 'building') {
        throw new Error(data.message || 'Failed to build vocabulary tree');
      }

      let sawLock = false;
      for (let waited = 0; waited < BUILD_WAIT_MS; ) {
        await sleep(BUILD_POLL_MS);
        waited += BUILD_POLL_MS;
        const poll = await fetch(`${apiBaseUrl}/vocab-tree/build/`);
        const state = await poll.json();
        if (state.status === 'exists') {
          return 'ready';
        }
        if (state.status === 'building') {
          sawLock = true;
        } else if (sawLock || waited >= BUILD_LOCK_GRACE_MS) {
          return 'failed';
        }
      }
      return 'timeout';
    } catch (err) {
      console.error('Failed to build vocabulary tree:', err);
      setError(`Build failed: ${err.message}`);
      return 'error';
    } finally {
      setBuilding(false);
    }
  }, [apiBaseUrl]);

  /**
   * Load top-level terms on mount
   */
  useEffect(() => {
    const loadTopLevel = async (retryAfterBuild = false) => {
      setLoading(prev => ({ ...prev, 0: true }));
      setError(null);
      setStillBuilding(false);
      setBuildFailed(false);
      try {
        const data = await fetchChildren('');
        const topLevel = {
          path: '',
          children: data.children,
          counts: data.counts,
          also: data.also || {},
          hasGrandchildren: data.has_grandchildren,
          isLeaf: data.is_leaf
        };
        // Store for later resets
        topLevelDataRef.current = topLevel;
        setLevels([topLevel]);
        setBuilding(false);
      } catch (err) {
        console.error('Failed to load vocabulary tree:', err);

        // If this is the first attempt, try to build the tree
        if (!retryAfterBuild) {
          console.log('Vocabulary tree not found, triggering build...');
          const outcome = await triggerBuild();
          if (outcome === 'ready') {
            // Retry loading after successful build
            loadTopLevel(true);
            return;
          }
          if (outcome === 'timeout') {
            setStillBuilding(true);
            setLoading(prev => ({ ...prev, 0: false }));
            return;
          }
          if (outcome === 'failed') {
            setBuildFailed(true);
            setLoading(prev => ({ ...prev, 0: false }));
            return;
          }
          // 'error': triggerBuild set the error message.
          setLoading(prev => ({ ...prev, 0: false }));
          return;
        }

        setError(err.message);
      }
      setLoading(prev => ({ ...prev, 0: false }));
    };
    loadTopLevel();
  }, [fetchChildren, triggerBuild, attempt]);

  /**
   * Handle selection at a specific level
   */
  const handleSelect = async (levelIndex, selectedOption) => {
    // Update selections array - truncate to current level
    const newSelections = selections.slice(0, levelIndex);

    if (selectedOption) {
      newSelections[levelIndex] = selectedOption.value;

      // Build path from all selections
      const path = newSelections.join('.');

      // Check if this selection has children
      const currentLevel = levels[levelIndex];
      const hasChildren = currentLevel.hasGrandchildren[selectedOption.value];

      if (hasChildren) {
        // Fetch children for the next level
        setLoading(prev => ({ ...prev, [levelIndex + 1]: true }));
        try {
          const data = await fetchChildren(path);
          // Update levels: keep levels up to current, add new level
          const newLevels = levels.slice(0, levelIndex + 1);
          newLevels.push({
            path: path,
            children: data.children,
            counts: data.counts,
            also: data.also || {},
            hasGrandchildren: data.has_grandchildren,
            isLeaf: data.is_leaf
          });
          setLevels(newLevels);
        } catch (err) {
          console.error('Failed to load children:', err);
          setError(err.message);
        }
        setLoading(prev => ({ ...prev, [levelIndex + 1]: false }));
      } else {
        // Leaf node - truncate levels to current
        setLevels(levels.slice(0, levelIndex + 1));
      }
    } else {
      // Selection cleared - truncate levels
      setLevels(levels.slice(0, levelIndex + 1));
    }

    setSelections(newSelections);
  };

  /**
   * Add the current selection path to the description
   */
  const handleAddToDescription = () => {
    if (selections.length === 0) return;

    // Join selections with spaces, add trailing semicolon
    const text = selections.join(' ') + '; ';

    // Call callback if provided
    if (onAddToDescription) {
      onAddToDescription(text);
    }

    // Insert into textarea at cursor position if ref provided
    if (descriptionRef?.current) {
      insertIntoDescription(descriptionRef.current, text);
    }

    // Clear selections after adding
    setSelections([]);
    // Reset to just top level using stored original data
    if (topLevelDataRef.current) {
      setLevels([topLevelDataRef.current]);
    }
  };

  /**
   * Clear all selections
   */
  const handleClear = () => {
    setSelections([]);
    // Reset to just top level using stored original data
    if (topLevelDataRef.current) {
      setLevels([topLevelDataRef.current]);
    }
  };

  /**
   * Convert children array to react-select options format.  The label
   * carries the document count and any synonyms renamed into the entry
   * ("Pileus (3,097; also: Cap)"); the value, which is what reaches the
   * description, carries neither.
   */
  const toOptions = (children, counts, also) =>
    children.map(child => {
      const parts = [];
      if (counts && counts[child] != null) {
        parts.push(countFormat.format(counts[child]));
      }
      const synonyms = (also && also[child]) || [];
      if (synonyms.length) parts.push(`also: ${synonyms.join(', ')}`);
      return {
        value: child,
        label: parts.length
          ? `${capitalize(child)} (${parts.join('; ')})`
          : capitalize(child),
      };
    });

  /**
   * Get placeholder text for a level
   */
  const getPlaceholder = (index, level) => {
    if (index === 0) return 'Select category...';
    if (level.isLeaf) return 'Select value...';
    return 'Select...';
  };

  if (building) {
    return (
      <div className="vocab-tree-widget vocab-tree-building">
        <p>Building vocabulary tree...</p>
        <p className="vocab-tree-building-note">This may take a moment.</p>
      </div>
    );
  }

  if (buildFailed && levels.length === 0) {
    return (
      <div className="vocab-tree-widget vocab-tree-error">
        <p>
          The vocabulary rebuild failed. The server's build log
          is <code>{BUILD_LOG}</code>.
        </p>
        <button onClick={() => setAttempt((n) => n + 1)}>Retry</button>
      </div>
    );
  }

  if (stillBuilding && levels.length === 0) {
    return (
      <div className="vocab-tree-widget vocab-tree-building">
        <p>
          The vocabulary menus are still being rebuilt; try again in a few
          minutes.
        </p>
        <button onClick={() => setAttempt((n) => n + 1)}>Retry</button>
      </div>
    );
  }

  if (error && levels.length === 0) {
    return (
      <div className="vocab-tree-widget vocab-tree-error">
        <p>Failed to load vocabulary: {error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="vocab-tree-widget">
      <div className="vocab-tree-header">
        <h4>Add Vocabulary Term</h4>
        {selections.length > 0 && (
          <button
            className="vocab-tree-clear-btn"
            onClick={handleClear}
            title="Clear selections"
          >
            Clear
          </button>
        )}
      </div>

      <div className="vocab-tree-selects">
        {levels.map((level, index) => (
          <div key={index} className="vocab-tree-level">
            <div className="vocab-tree-level-indicator">
              {index > 0 && <span className="vocab-tree-arrow">&#8627;</span>}
            </div>
            <Select
              options={toOptions(level.children, level.counts, level.also)}
              value={selections[index]
                ? { value: selections[index], label: capitalize(selections[index]) }
                : null
              }
              onChange={(option) => handleSelect(index, option)}
              isLoading={loading[index]}
              isClearable
              isSearchable
              placeholder={getPlaceholder(index, level)}
              styles={selectStyles}
              className="vocab-tree-select"
              classNamePrefix="vocab-select"
              noOptionsMessage={() => 'No matching terms'}
            />
          </div>
        ))}
      </div>

      {selections.length > 0 && (
        <div className="vocab-tree-preview">
          <span className="vocab-tree-preview-label">Preview:</span>
          <code className="vocab-tree-preview-text">{selections.join(' ')};</code>
        </div>
      )}

      <div className="vocab-tree-actions">
        <button
          className="vocab-tree-add-btn"
          onClick={handleAddToDescription}
          disabled={selections.length === 0}
        >
          Add to Description
        </button>
      </div>
    </div>
  );
};

export default VocabTreeWidget;
