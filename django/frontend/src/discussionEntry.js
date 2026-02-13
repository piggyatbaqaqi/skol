/**
 * Entry point for DiscussionWidget bundle
 *
 * Initializes the DiscussionWidget on elements with
 * data-discussion-widget attribute.
 *
 * Usage in HTML:
 *   <div data-discussion-widget
 *        data-api-base-url="/skol/api"
 *        data-collection-id="123456789">
 *   </div>
 *
 *   <script src="{% static 'js/discussion.bundle.js' %}"></script>
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import DiscussionWidget from './DiscussionWidget';

function initDiscussionWidgets() {
  const containers = document.querySelectorAll('[data-discussion-widget]');

  containers.forEach((container) => {
    // Skip if already initialized
    if (container.dataset.initialized === 'true') {
      return;
    }

    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const collectionId = container.dataset.collectionId || '';

    // Mark as initialized to prevent re-initialization
    container.dataset.initialized = 'true';

    const root = createRoot(container);
    root.render(
      <DiscussionWidget
        apiBaseUrl={apiBaseUrl}
        initialCollectionId={collectionId ? Number(collectionId) : null}
        container={container}
      />
    );
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDiscussionWidgets);
} else {
  initDiscussionWidgets();
}

// Listen for dynamic initialization (e.g., after collection select loads)
document.addEventListener('discussion-widget-init', initDiscussionWidgets);

export { DiscussionWidget };
export default DiscussionWidget;
