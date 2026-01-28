/**
 * FungariumSelect - Searchable dropdown for selecting fungaria/herbaria
 *
 * A React component using react-select that loads fungaria from the API
 * and allows searching/filtering by code or organization name.
 */
import React, { useState, useEffect, useCallback } from 'react';
import Select from 'react-select';
import './FungariumSelect.css';

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
    padding: '8px 12px',
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
 * Custom option component to show code and organization
 */
const FungariumOption = ({ data, innerRef, innerProps, isFocused, isSelected }) => {
  return (
    <div
      ref={innerRef}
      {...innerProps}
      className={`fungarium-option ${isFocused ? 'focused' : ''} ${isSelected ? 'selected' : ''}`}
    >
      <span className="fungarium-code">{data.code}</span>
      <span className="fungarium-org">{data.organization}</span>
      {data.location && <span className="fungarium-location">{data.location}</span>}
    </div>
  );
};

/**
 * Custom single value display
 */
const FungariumSingleValue = ({ data }) => {
  return (
    <div className="fungarium-single-value">
      <span className="fungarium-code">{data.code}</span>
      <span className="fungarium-org">{data.organization}</span>
    </div>
  );
};

/**
 * FungariumSelect Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {Function} props.onChange - Callback when selection changes (receives code)
 * @param {string} props.value - Currently selected fungarium code
 * @param {string} props.placeholder - Placeholder text
 * @param {boolean} props.isDisabled - Whether the select is disabled
 */
const FungariumSelect = ({
  apiBaseUrl = '/api',
  onChange,
  value,
  placeholder = 'Search fungaria...',
  isDisabled = false
}) => {
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch fungaria from API
   */
  const fetchFungaria = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/fungaria/`, {
        credentials: 'same-origin'
      });

      if (!response.ok) {
        throw new Error('Failed to load fungaria');
      }

      const data = await response.json();
      const fungaria = data.fungaria || [];

      // Convert to react-select options format
      const opts = fungaria.map(f => ({
        value: f.code,
        label: `${f.code} - ${f.organization}`,
        code: f.code,
        organization: f.organization,
        location: f.location,
        num_fungi: f.num_fungi,
        collection_url: f.collection_url,
        web_url: f.web_url
      }));

      setOptions(opts);
    } catch (err) {
      console.error('Error loading fungaria:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  // Load fungaria on mount
  useEffect(() => {
    fetchFungaria();
  }, [fetchFungaria]);

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
    ? options.find(opt => opt.value === value) || null
    : null;

  /**
   * Custom filter function to search by code and organization
   */
  const filterOption = (option, searchText) => {
    if (!searchText) return true;
    const search = searchText.toLowerCase();
    return (
      option.data.code.toLowerCase().includes(search) ||
      option.data.organization.toLowerCase().includes(search) ||
      (option.data.location && option.data.location.toLowerCase().includes(search))
    );
  };

  if (error) {
    return (
      <div className="fungarium-select-error">
        Failed to load fungaria. <button onClick={fetchFungaria}>Retry</button>
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
      isClearable
      isSearchable
      placeholder={placeholder}
      styles={selectStyles}
      className="fungarium-select"
      classNamePrefix="fungarium"
      components={{
        Option: FungariumOption,
        SingleValue: FungariumSingleValue
      }}
      filterOption={filterOption}
      noOptionsMessage={() => 'No matching fungaria'}
      loadingMessage={() => 'Loading fungaria...'}
    />
  );
};

export default FungariumSelect;
