/**
 * Entry point for VocabTreeWidget bundle
 *
 * This file initializes the VocabTreeWidget on elements with
 * data-vocab-tree-widget attribute.
 *
 * Usage in HTML:
 *   <div data-vocab-tree-widget
 *        data-api-base-url="/api"
 *        data-description-target="#description-textarea">
 *   </div>
 *
 *   <script src="{% static 'js/vocab-tree.bundle.js' %}"></script>
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import VocabTreeWidget from './VocabTreeWidget';

/**
 * Initialize VocabTreeWidget on all matching elements
 */
function initVocabTreeWidgets() {
  const containers = document.querySelectorAll('[data-vocab-tree-widget]');

  containers.forEach((container) => {
    // Get configuration from data attributes
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
      // Dispatch custom event for integration with other frameworks
      const event = new CustomEvent('vocab-tree-add', {
        detail: { text },
        bubbles: true
      });
      container.dispatchEvent(event);
    };

    // Create React root and render
    const root = createRoot(container);
    root.render(
      <VocabTreeWidget
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
  document.addEventListener('DOMContentLoaded', initVocabTreeWidgets);
} else {
  initVocabTreeWidgets();
}

// Export for programmatic use
export { VocabTreeWidget };
export default VocabTreeWidget;
