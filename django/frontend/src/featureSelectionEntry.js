/**
 * Entry point for FeatureSelectionWidget bundle
 *
 * This file initializes the FeatureSelectionWidget on elements with
 * data-feature-selection-widget attribute.
 *
 * Usage in HTML:
 *   <div data-feature-selection-widget
 *        data-api-base-url="/skol/api"
 *        data-description-target="#prompt"
 *        data-collection-source="#collectionSelect">
 *   </div>
 *
 *   <script src="{% static 'js/feature-selection.bundle.js' %}"></script>
 */
import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import FeatureSelectionWidget from './FeatureSelectionWidget';

/**
 * Wrapper component that listens for collection changes
 * and passes the active collectionId to FeatureSelectionWidget.
 */
const FeatureSelectionWrapper = ({
  apiBaseUrl,
  version,
  descriptionRef,
  onAddToDescription,
  collectionSourceSelector,
}) => {
  const [collectionId, setCollectionId] = useState(null);

  useEffect(() => {
    // Listen for collection-changed custom events dispatched by the page
    const handleCollectionChanged = (event) => {
      const id = event.detail?.collection_id || event.detail?.collectionId || null;
      setCollectionId(id);
    };
    document.addEventListener('collection-changed', handleCollectionChanged);

    // Also try to read the initial value from the source element
    if (collectionSourceSelector) {
      const el = document.querySelector(collectionSourceSelector);
      if (el && el.value) {
        setCollectionId(el.value);
      }
    }

    return () => {
      document.removeEventListener('collection-changed', handleCollectionChanged);
    };
  }, [collectionSourceSelector]);

  return (
    <FeatureSelectionWidget
      apiBaseUrl={apiBaseUrl}
      version={version}
      descriptionRef={descriptionRef}
      onAddToDescription={onAddToDescription}
      collectionId={collectionId}
    />
  );
};

/**
 * Initialize FeatureSelectionWidget on all matching elements
 */
function initFeatureSelectionWidgets() {
  const containers = document.querySelectorAll('[data-feature-selection-widget]');

  containers.forEach((container) => {
    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const version = container.dataset.version || null;
    const descriptionTargetSelector = container.dataset.descriptionTarget;
    const collectionSourceSelector = container.dataset.collectionSource || null;

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
      <FeatureSelectionWrapper
        apiBaseUrl={apiBaseUrl}
        version={version}
        descriptionRef={descriptionRef}
        onAddToDescription={handleAddToDescription}
        collectionSourceSelector={collectionSourceSelector}
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
