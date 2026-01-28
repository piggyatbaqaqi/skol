/**
 * Entry point for IdentifierTypeSelect bundle
 *
 * This file initializes the IdentifierTypeSelect on elements with
 * data-identifier-type-select attribute.
 *
 * Usage in HTML:
 *   <div data-identifier-type-select
 *        data-api-base-url="/api"
 *        data-value-input="#identifier-type-input"
 *        data-placeholder="Select type..."
 *        data-on-change="onIdentifierTypeChange">
 *   </div>
 *
 *   <script src="{% static 'js/identifier-type-select.bundle.js' %}"></script>
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import IdentifierTypeSelect from './IdentifierTypeSelect';

/**
 * Wrapper component that handles state and input synchronization
 */
const IdentifierTypeSelectWrapper = ({
  apiBaseUrl,
  valueInput,
  placeholder,
  initialValue,
  onChangeCallbackName
}) => {
  const [value, setValue] = React.useState(initialValue || '');

  const handleChange = (newValue) => {
    setValue(newValue);

    // Update hidden input if specified
    if (valueInput) {
      const input = document.querySelector(valueInput);
      if (input) {
        input.value = newValue;
        // Dispatch change event for form validation
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    }

    // Call named callback function if provided
    if (onChangeCallbackName && typeof window[onChangeCallbackName] === 'function') {
      window[onChangeCallbackName](newValue);
    }

    // Dispatch custom event for integration
    const event = new CustomEvent('identifier-type-select-change', {
      detail: { value: newValue },
      bubbles: true
    });
    document.dispatchEvent(event);
  };

  // Listen for external value changes
  React.useEffect(() => {
    if (valueInput) {
      const input = document.querySelector(valueInput);
      if (input) {
        // Set initial value from input
        if (input.value && !initialValue) {
          setValue(input.value);
        }

        // Listen for external changes
        const handleInputChange = () => {
          setValue(input.value);
        };
        input.addEventListener('change', handleInputChange);
        return () => input.removeEventListener('change', handleInputChange);
      }
    }
  }, [valueInput, initialValue]);

  return (
    <IdentifierTypeSelect
      apiBaseUrl={apiBaseUrl}
      value={value}
      onChange={handleChange}
      placeholder={placeholder}
    />
  );
};

/**
 * Initialize IdentifierTypeSelect on all matching elements
 */
function initIdentifierTypeSelects() {
  const containers = document.querySelectorAll('[data-identifier-type-select]');

  containers.forEach((container) => {
    // Skip if already initialized
    if (container.dataset.identifierTypeInitialized === 'true') {
      return;
    }

    // Get configuration from data attributes
    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const valueInputSelector = container.dataset.valueInput;
    const placeholder = container.dataset.placeholder || 'Select identifier type...';
    const initialValue = container.dataset.value || '';
    const onChangeCallbackName = container.dataset.onChange;

    // Create React root and render
    const root = createRoot(container);
    root.render(
      <IdentifierTypeSelectWrapper
        apiBaseUrl={apiBaseUrl}
        valueInput={valueInputSelector}
        placeholder={placeholder}
        initialValue={initialValue}
        onChangeCallbackName={onChangeCallbackName}
      />
    );

    // Mark as initialized
    container.dataset.identifierTypeInitialized = 'true';
  });
}

/**
 * Re-initialize function for dynamically added elements
 */
window.initIdentifierTypeSelects = initIdentifierTypeSelects;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initIdentifierTypeSelects);
} else {
  initIdentifierTypeSelects();
}

// Export for programmatic use
export { IdentifierTypeSelect };
export default IdentifierTypeSelect;
