/**
 * Entry point for CollectionSelect bundle
 *
 * This file initializes the CollectionSelect on elements with
 * data-collection-select attribute.
 *
 * Usage in HTML:
 *   <div data-collection-select
 *        data-api-base-url="/api"
 *        data-value-input="#collection-id-input"
 *        data-placeholder="Select collection..."
 *        data-on-change="onCollectionChange">
 *   </div>
 *
 *   <script src="{% static 'js/collection-select.bundle.js' %}"></script>
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import CollectionSelect from './CollectionSelect';

/**
 * Wrapper component that handles state and input synchronization
 */
const CollectionSelectWrapper = ({
  apiBaseUrl,
  valueInput,
  placeholder,
  initialValue,
  onChangeCallbackName,
  allowNone
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
    const event = new CustomEvent('collection-select-change', {
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
    <CollectionSelect
      apiBaseUrl={apiBaseUrl}
      value={value}
      onChange={handleChange}
      placeholder={placeholder}
      allowNone={allowNone}
    />
  );
};

/**
 * Initialize CollectionSelect on all matching elements
 */
function initCollectionSelects() {
  const containers = document.querySelectorAll('[data-collection-select]');

  containers.forEach((container) => {
    // Skip if already initialized
    if (container.dataset.collectionInitialized === 'true') {
      return;
    }

    // Get configuration from data attributes
    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const valueInputSelector = container.dataset.valueInput;
    const placeholder = container.dataset.placeholder || 'Select collection...';
    const initialValue = container.dataset.value || '';
    const onChangeCallbackName = container.dataset.onChange;
    const allowNone = container.dataset.allowNone !== 'false';

    // Create React root and render
    const root = createRoot(container);
    root.render(
      <CollectionSelectWrapper
        apiBaseUrl={apiBaseUrl}
        valueInput={valueInputSelector}
        placeholder={placeholder}
        initialValue={initialValue}
        onChangeCallbackName={onChangeCallbackName}
        allowNone={allowNone}
      />
    );

    // Mark as initialized
    container.dataset.collectionInitialized = 'true';
  });
}

/**
 * Re-initialize function for dynamically added elements
 */
window.initCollectionSelects = initCollectionSelects;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCollectionSelects);
} else {
  initCollectionSelects();
}

// Export for programmatic use
export { CollectionSelect };
export default CollectionSelect;
