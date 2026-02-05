/**
 * Entry point for FeatureSelectionWidget bundle
 *
 * This file initializes the FeatureSelectionWidget on elements with
 * data-feature-selection-widget attribute.
 *
 * Usage in HTML:
 *   <div data-feature-selection-widget
 *        data-api-base-url="/skol/api"
 *        data-description-target="#prompt">
 *   </div>
 *
 *   <script src="{% static 'js/feature-selection.bundle.js' %}"></script>
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import FeatureSelectionWidget from './FeatureSelectionWidget';

/**
 * Initialize FeatureSelectionWidget on all matching elements
 */
function initFeatureSelectionWidgets() {
  const containers = document.querySelectorAll('[data-feature-selection-widget]');

  containers.forEach((container) => {
    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const version = container.dataset.version || null;
    const descriptionTargetSelector = container.dataset.descriptionTarget;

    // Find the description textarea if specified
    let descriptionRef = null;
    if (descriptionTargetSelector) {
      const textarea = document.querySelector(descriptionTargetSelector);
      if (textarea) {
        descriptionRef = { current: textarea };
      }
    }

    // Callback to handle adding text to description
    const handleAddToDescription = (text) => {
      const event = new CustomEvent('feature-selection-add', {
        detail: { text },
        bubbles: true,
      });
      container.dispatchEvent(event);
    };

    const root = createRoot(container);
    root.render(
      <FeatureSelectionWidget
        apiBaseUrl={apiBaseUrl}
        version={version}
        descriptionRef={descriptionRef}
        onAddToDescription={handleAddToDescription}
      />
    );
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initFeatureSelectionWidgets);
} else {
  initFeatureSelectionWidgets();
}

export { FeatureSelectionWidget };
export default FeatureSelectionWidget;
