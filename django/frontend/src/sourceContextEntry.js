/**
 * Entry point for SourceContextViewer bundle
 *
 * This file initializes the SourceContextViewer on elements with
 * data-source-context-viewer attribute.
 *
 * Usage in HTML:
 *   <div data-source-context-viewer
 *        data-taxa-id="taxon_xxx"
 *        data-field="description"
 *        data-api-base-url="/skol/api"
 *        data-context-chars="500"
 *        data-taxa-db="skol_taxa_dev"
 *        data-original-text="The original text to display">
 *   </div>
 *
 *   <script src="{% static 'js/source-context.bundle.js' %}"></script>
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import SourceContextViewer from './SourceContextViewer';

/**
 * Initialize SourceContextViewer on all matching elements
 */
function initSourceContextViewers() {
  const containers = document.querySelectorAll('[data-source-context-viewer]');

  containers.forEach((container) => {
    // Skip if already initialized
    if (container.dataset.initialized === 'true') {
      return;
    }

    const taxaId = container.dataset.taxaId;
    const field = container.dataset.field || 'description';
    const apiBaseUrl = container.dataset.apiBaseUrl || '/api';
    const contextChars = parseInt(container.dataset.contextChars, 10) || 500;
    const taxaDb = container.dataset.taxaDb || 'skol_taxa_dev';
    const originalText = container.dataset.originalText || '';

    if (!taxaId) {
      console.error('SourceContextViewer: missing data-taxa-id attribute');
      return;
    }

    // Mark as initialized to prevent re-initialization
    container.dataset.initialized = 'true';

    const root = createRoot(container);
    root.render(
      <SourceContextViewer
        taxaId={taxaId}
        field={field}
        apiBaseUrl={apiBaseUrl}
        contextChars={contextChars}
        taxaDb={taxaDb}
        originalText={originalText}
      />
    );
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSourceContextViewers);
} else {
  initSourceContextViewers();
}

// Also listen for dynamic content (e.g., search results loading)
document.addEventListener('source-context-init', initSourceContextViewers);

export { SourceContextViewer };
export default SourceContextViewer;
