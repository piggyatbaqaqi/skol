/**
 * IdentifierTypeSelect - Searchable dropdown for selecting identifier types
 *
 * A React component using react-select for selecting external identifier types
 * (iNaturalist, Mushroom Observer, GenBank, Fungarium, etc.)
 */
import React, { useState, useEffect, useCallback } from 'react';
import Select from 'react-select';
import './IdentifierTypeSelect.css';

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
 * IdentifierTypeSelect Component
 *
 * @param {Object} props
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {Function} props.onChange - Callback when selection changes (receives code)
 * @param {string} props.value - Currently selected identifier type code
 * @param {string} props.placeholder - Placeholder text
 * @param {boolean} props.isDisabled - Whether the select is disabled
 */
const IdentifierTypeSelect = ({
  apiBaseUrl = '/api',
  onChange,
  value,
  placeholder = 'Select identifier type...',
  isDisabled = false
}) => {
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch identifier types from API
   */
  const fetchIdentifierTypes = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/identifier-types/`, {
        credentials: 'same-origin'
      });

      if (!response.ok) {
        throw new Error('Failed to load identifier types');
      }

      const data = await response.json();
      const types = data.identifier_types || [];

      // Convert to react-select options format
      const opts = types.map(t => ({
        value: t.code,
        label: t.name,
        code: t.code,
        name: t.name,
        description: t.description
      }));

      setOptions(opts);
    } catch (err) {
      console.error('Error loading identifier types:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  // Load identifier types on mount
  useEffect(() => {
    fetchIdentifierTypes();
  }, [fetchIdentifierTypes]);

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

  if (error) {
    return (
      <div className="identifier-type-select-error">
        Failed to load types. <button onClick={fetchIdentifierTypes}>Retry</button>
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
      className="identifier-type-select"
      classNamePrefix="id-type"
      noOptionsMessage={() => 'No matching types'}
      loadingMessage={() => 'Loading...'}
    />
  );
};

export default IdentifierTypeSelect;
