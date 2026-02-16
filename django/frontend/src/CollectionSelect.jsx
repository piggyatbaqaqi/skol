/**
 * CollectionSelect - Searchable dropdown for selecting user collections
 *
 * A React component using react-select for selecting from user's collections.
 */
import React, { useState, useEffect, useCallback } from 'react';
import Select from 'react-select';
import './CollectionSelect.css';

/**
 * Custom styles for react-select to match the SKOL theme
 */
const selectStyles = {
  control: (base, state) => ({
    ...base,
    minHeight: '38px',
    borderColor: state.isFocused ? '#667eea' : '#e0e0e0',
    boxShadow: state.isFocused ? '0 0 0 1px #667eea' : 'none',
    '&:hover': {
      borderColor: '#667eea'
    }
  }),
  option: (base, state) => ({
    ...base,
    backgroundColor: state.isSelected
      ? '#667eea'
      : state.isFocused
        ? '#f0f4ff'
        : 'white',
    color: state.isSelected ? 'white' : '#333',
    cursor: 'pointer',
    padding: '10px 12px',
    '&:active': {
      backgroundColor: '#667eea'
    }
  }),
  menu: (base) => ({
    ...base,
    zIndex: 9999
  }),
  placeholder: (base) => ({
    ...base,
    color: '#999'
  }),
  singleValue: (base) => ({
    ...base,
    color: '#333'
  }),
  input: (base) => ({
    ...base,
    color: '#333'
  })
};

/**
 * Custom option component to show collection name and ID
 */
const CollectionOption = ({ data, innerRef, innerProps, isFocused, isSelected }) => {
  return (
    <div
      ref={innerRef}
      {...innerProps}
      className={`collection-option ${isFocused ? 'focused' : ''} ${isSelected ? 'selected' : ''}`}
      style={data.hidden ? { opacity: 0.5 } : {}}
    >
      <span className="collection-name">{data.hidden ? '\u2298 ' : ''}{data.name}</span>
      <span className="collection-id">#{data.collectionId}</span>
    </div>
  );
};

/**
 * CollectionSelect Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {Function} props.onChange - Callback when selection changes (receives collection_id)
 * @param {string} props.value - Currently selected collection ID
 * @param {string} props.placeholder - Placeholder text
 * @param {boolean} props.isDisabled - Whether the select is disabled
 * @param {boolean} props.allowNone - Whether to allow "No collection" option
 */
const CollectionSelect = ({
  apiBaseUrl = '/api',
  onChange,
  value,
  placeholder = 'Select collection...',
  isDisabled = false,
  allowNone = true
}) => {
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch collections from API
   */
  const fetchCollections = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/collections/`, {
        credentials: 'same-origin'
      });

      if (!response.ok) {
        throw new Error('Failed to load collections');
      }

      const data = await response.json();
      const collections = data.collections || [];

      // Convert to react-select options format
      const opts = collections.map(c => ({
        value: String(c.collection_id),
        label: c.name,
        name: c.name,
        collectionId: c.collection_id,
        searchCount: c.search_count,
        identifierCount: c.identifier_count,
        hidden: c.hidden || false
      }));

      setOptions(opts);
    } catch (err) {
      console.error('Error loading collections:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  // Load collections on mount
  useEffect(() => {
    fetchCollections();
  }, [fetchCollections]);

  // Expose refresh function globally
  useEffect(() => {
    window.refreshCollectionSelect = fetchCollections;
  }, [fetchCollections]);

  /**
   * Handle selection change
   */
  const handleChange = (selected) => {
    if (onChange) {
      onChange(selected ? selected.value : '');
    }
  };

  /**
   * Find the currently selected option
   */
  const selectedOption = value
    ? options.find(opt => opt.value === String(value)) || null
    : null;

  if (error) {
    return (
      <div className="collection-select-error">
        Failed to load collections. <button onClick={fetchCollections}>Retry</button>
      </div>
    );
  }

  return (
    <Select
      options={options}
      value={selectedOption}
      onChange={handleChange}
      isLoading={loading}
      isDisabled={isDisabled}
      isClearable={allowNone}
      isSearchable
      placeholder={placeholder}
      styles={selectStyles}
      className="collection-select"
      classNamePrefix="collection"
      components={{
        Option: CollectionOption
      }}
      noOptionsMessage={() => 'No collections found'}
      loadingMessage={() => 'Loading collections...'}
    />
  );
};

export default CollectionSelect;
