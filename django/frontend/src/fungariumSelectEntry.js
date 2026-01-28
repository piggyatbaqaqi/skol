/**
 * Entry point for FungariumSelect bundle
 *
 * This file initializes the FungariumSelect on elements with
 * data-fungarium-select attribute.
 *
 * Usage in HTML:
 *   <div data-fungarium-select
 *        data-api-base-url="/api"
 *        data-value-input="#fungarium-code-input"
 *        data-placeholder="Search fungaria...">
 *   </div>
 *
 *   <script src="{% static 'js/fungarium-select.bundle.js' %}"></script>
 *
 * The component will update the hidden input specified by data-value-input
 * with the selected fungarium code.
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import FungariumSelect from './FungariumSelect';

/**
 * Wrapper component that handles state and input synchronization
 */
const FungariumSelectWrapper = ({
  apiBaseUrl,
  valueInput,
  placeholder,
  initialValue,
  onChangeCallback
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

    // Call external callback if provided
    if (onChangeCallback) {
      onChangeCallback(newValue);
    }

    // Dispatch custom event for integration
    const event = new CustomEvent('fungarium-select-change', {
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
    <FungariumSelect
      apiBaseUrl={apiBaseUrl}
      value={value}
      onChange={handleChange}
      placeholder={placeholder}
    />
  );
};

/**
 * Initialize FungariumSelect on all matching elements
 */
function initFungariumSelects() {
  const containers = document.querySelectorAll('[data-fungarium-select]');

  containers.forEach((container) => {
    // Skip if already initialized
    if (container.dataset.fungariumInitialized === 'true') {
      return;
    }

    // Get configuration from data attributes
    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const valueInputSelector = container.dataset.valueInput;
    const placeholder = container.dataset.placeholder || 'Search fungaria...';
    const initialValue = container.dataset.value || '';

    // Create React root and render
    const root = createRoot(container);
    root.render(
      <FungariumSelectWrapper
        apiBaseUrl={apiBaseUrl}
        valueInput={valueInputSelector}
        placeholder={placeholder}
        initialValue={initialValue}
      />
    );

    // Mark as initialized
    container.dataset.fungariumInitialized = 'true';
  });
}

/**
 * Re-initialize function for dynamically added elements
 */
window.initFungariumSelects = initFungariumSelects;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initFungariumSelects);
} else {
  initFungariumSelects();
}

// Export for programmatic use
export { FungariumSelect };
export default FungariumSelect;
