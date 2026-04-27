/**
 * Entry point for ProjectSelect bundle
 *
 * Initializes ProjectSelect on elements with [data-project-select].
 *
 * Usage in HTML:
 *   <div data-project-select
 *        data-api-base-url="/api"
 *        data-value-input="#project-slugs-input"
 *        data-placeholder="Filter by project..."
 *        data-is-multi="true">
 *   </div>
 *
 *   <script src="{% static 'js/project-select.bundle.js' %}"></script>
 *
 * The component dispatches a "project-select-change" CustomEvent on document
 * with detail: { value: [slug, ...] } (or string when is-multi="false").
 *
 * It also updates the hidden input named by data-value-input with a
 * comma-separated list of selected slugs.
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import ProjectSelect from './ProjectSelect';

const ProjectSelectWrapper = ({
  apiBaseUrl,
  valueInput,
  placeholder,
  initialValue,
  isMulti,
  onChangeCallbackName,
}) => {
  const [value, setValue] = React.useState(
    isMulti
      ? (initialValue ? initialValue.split(',').map((s) => s.trim()).filter(Boolean) : [])
      : (initialValue || '')
  );

  const handleChange = (newValue) => {
    setValue(newValue);

    // Sync hidden input
    if (valueInput) {
      const input = document.querySelector(valueInput);
      if (input) {
        input.value = Array.isArray(newValue) ? newValue.join(',') : (newValue || '');
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    }

    // Named window callback
    if (onChangeCallbackName && typeof window[onChangeCallbackName] === 'function') {
      window[onChangeCallbackName](newValue);
    }

    // Custom event
    document.dispatchEvent(
      new CustomEvent('project-select-change', {
        detail: { value: newValue },
        bubbles: true,
      })
    );
  };

  return (
    <ProjectSelect
      apiBaseUrl={apiBaseUrl}
      value={value}
      onChange={handleChange}
      placeholder={placeholder}
      isMulti={isMulti}
    />
  );
};

function initProjectSelects() {
  const containers = document.querySelectorAll('[data-project-select]');
  containers.forEach((container) => {
    if (container.dataset.projectInitialized === 'true') return;

    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const valueInputSelector = container.dataset.valueInput;
    const placeholder = container.dataset.placeholder || 'Filter by project...';
    const initialValue = container.dataset.value || '';
    const isMulti = container.dataset.isMulti !== 'false';
    const onChangeCallbackName = container.dataset.onChange;

    const root = createRoot(container);
    root.render(
      <ProjectSelectWrapper
        apiBaseUrl={apiBaseUrl}
        valueInput={valueInputSelector}
        placeholder={placeholder}
        initialValue={initialValue}
        isMulti={isMulti}
        onChangeCallbackName={onChangeCallbackName}
      />
    );

    container.dataset.projectInitialized = 'true';
  });
}

window.initProjectSelects = initProjectSelects;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initProjectSelects);
} else {
  initProjectSelects();
}

export { ProjectSelect };
export default ProjectSelect;
