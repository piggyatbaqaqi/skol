/**
 * ProjectSelect - Searchable multi-select dropdown for projects
 *
 * Loads all projects from GET /api/projects/ (optionally filtered by ?q=)
 * and lets the user pick one or more projects (for filtering collections).
 * Uses react-select per SKOL convention.
 */
import React, { useState, useEffect, useCallback } from 'react';
import Select from 'react-select';
import './ProjectSelect.css';

const selectStyles = {
  control: (base, state) => ({
    ...base,
    minHeight: '38px',
    borderColor: state.isFocused ? '#667eea' : '#e0e0e0',
    boxShadow: state.isFocused ? '0 0 0 1px #667eea' : 'none',
    '&:hover': { borderColor: '#667eea' },
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
    '&:active': { backgroundColor: '#667eea' },
  }),
  menu: (base) => ({ ...base, zIndex: 9999 }),
  placeholder: (base) => ({ ...base, color: '#999' }),
  multiValue: (base) => ({
    ...base,
    backgroundColor: '#f0f4ff',
    borderRadius: '4px',
  }),
  multiValueLabel: (base) => ({ ...base, color: '#667eea', fontWeight: 500 }),
  multiValueRemove: (base) => ({
    ...base,
    color: '#667eea',
    '&:hover': { backgroundColor: '#667eea', color: 'white' },
  }),
};

const ProjectOption = ({ data, innerRef, innerProps, isFocused, isSelected }) => (
  <div
    ref={innerRef}
    {...innerProps}
    className={`project-option ${isFocused ? 'focused' : ''} ${isSelected ? 'selected' : ''}`}
  >
    <span className="project-name">{data.name}</span>
    <span className="project-slug">{data.namespacedSlug}</span>
  </div>
);

/**
 * ProjectSelect Component
 *
 * @param {Object}   props
 * @param {string}   props.apiBaseUrl   - Base URL for API calls (default: '/api')
 * @param {Function} props.onChange     - Called with array of namespaced slugs
 * @param {string[]} props.value        - Currently selected namespaced slugs
 * @param {string}   props.placeholder  - Placeholder text
 * @param {boolean}  props.isDisabled   - Whether the select is disabled
 * @param {boolean}  props.isMulti      - Allow multiple selections (default: true)
 */
const ProjectSelect = ({
  apiBaseUrl = '/api',
  onChange,
  value = [],
  placeholder = 'Filter by project...',
  isDisabled = false,
  isMulti = true,
}) => {
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchProjects = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/projects/`, {
        credentials: 'same-origin',
      });
      if (!response.ok) throw new Error('Failed to load projects');
      const data = await response.json();
      const projects = data.projects || [];
      setOptions(
        projects.map((p) => ({
          value: p.namespaced_slug,
          label: p.name,
          name: p.name,
          namespacedSlug: p.namespaced_slug,
          collectionCount: p.collection_count,
        }))
      );
    } catch (err) {
      console.error('Error loading projects:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  useEffect(() => {
    window.refreshProjectSelect = fetchProjects;
  }, [fetchProjects]);

  const handleChange = (selected) => {
    if (onChange) {
      if (isMulti) {
        onChange(selected ? selected.map((s) => s.value) : []);
      } else {
        onChange(selected ? selected.value : '');
      }
    }
  };

  const selectedOptions = isMulti
    ? options.filter((o) => value.includes(o.value))
    : options.find((o) => o.value === value) || null;

  if (error) {
    return (
      <div className="project-select-error">
        Failed to load projects. <button onClick={fetchProjects}>Retry</button>
      </div>
    );
  }

  return (
    <Select
      options={options}
      value={selectedOptions}
      onChange={handleChange}
      isLoading={loading}
      isDisabled={isDisabled}
      isClearable
      isSearchable
      isMulti={isMulti}
      placeholder={placeholder}
      styles={selectStyles}
      className="project-select"
      classNamePrefix="project"
      components={{ Option: ProjectOption }}
      noOptionsMessage={() => 'No projects found'}
      loadingMessage={() => 'Loading projects...'}
    />
  );
};

export default ProjectSelect;
