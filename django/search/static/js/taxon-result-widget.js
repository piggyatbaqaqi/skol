/**
 * TaxonResultWidget - A reusable widget for displaying taxon search results
 *
 * This widget can render a taxon result card either from:
 * 1. Full result data (from search API)
 * 2. A taxon_id (fetches data from /api/taxa/<id>/)
 * 3. Cached reference data (similarity, title, taxon_id)
 */

const TaxonResultWidget = (function() {
    'use strict';

    // Cache for fetched taxon data
    const taxonCache = new Map();

    // API base URL (can be overridden)
    let apiBase = '';

    /**
     * Initialize the widget with configuration
     * @param {Object} config - Configuration object
     * @param {string} config.apiBase - Base URL for API calls
     */
    function init(config = {}) {
        apiBase = config.apiBase || '';
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped HTML
     */
    function escapeHtml(text) {
        if (text === null || text === undefined) return '';
        const div = document.createElement('div');
        div.textContent = String(text);
        return div.innerHTML;
    }

    /**
     * Fetch taxon data from the API
     * @param {string} taxonId - The taxon ID to fetch
     * @returns {Promise<Object>} The taxon data
     */
    async function fetchTaxon(taxonId) {
        if (!taxonId) {
            throw new Error('taxonId is required');
        }

        // Check cache first
        if (taxonCache.has(taxonId)) {
            return taxonCache.get(taxonId);
        }

        const response = await fetch(`${apiBase}/api/taxa/${encodeURIComponent(taxonId)}/`, {
            credentials: 'same-origin'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `Failed to fetch taxon: ${taxonId}`);
        }

        const data = await response.json();
        taxonCache.set(taxonId, data);
        return data;
    }

    /**
     * Build PDF viewer URL from result data
     * @param {Object} result - Result object with PDF metadata
     * @returns {Object} Object with pdfUrl and pageFragment
     */
    function buildPdfUrl(result) {
        let pdfUrl = null;
        let pageFragment = '';

        // Determine page fragment for direct linking
        if (result.PDFLabel) {
            pageFragment = `#page=${encodeURIComponent(result.PDFLabel)}`;
        } else if (result.PDFPage) {
            pageFragment = `#page=${result.PDFPage}`;
        }

        // Build PDF URL if source metadata is available
        // Links to /pdf/ viewer page (not /api/pdf/ which is the raw PDF endpoint)
        if (result.PDFDbName && result.PDFDocId) {
            pdfUrl = `${apiBase}/pdf/?db=${encodeURIComponent(result.PDFDbName)}&doc_id=${encodeURIComponent(result.PDFDocId)}`;
            if (result.PDFPage) {
                pdfUrl += `&page=${result.PDFPage}`;
            }
        }

        return { pdfUrl, pageFragment };
    }

    /**
     * Create a result card element from taxon data
     * @param {Object} result - The result data
     * @param {Object} options - Rendering options
     * @param {number} options.similarity - Override similarity score (0-1)
     * @param {number} options.index - Index for JSON toggle (optional)
     * @param {boolean} options.showJson - Whether to show JSON toggle button
     * @param {boolean} options.compact - Whether to use compact mode
     * @returns {HTMLElement} The card element
     */
    function createCard(result, options = {}) {
        const card = document.createElement('div');
        card.className = options.compact ? 'taxon-result-card compact' : 'taxon-result-card';

        // Use provided similarity or from result
        const similarity = options.similarity !== undefined ? options.similarity : result.Similarity;
        const similarityPercent = similarity !== undefined && similarity !== null
            ? (similarity * 100).toFixed(1)
            : null;

        // Build PDF URL
        const { pdfUrl, pageFragment } = buildPdfUrl(result);

        // Build PDF link HTML
        let pdfLink = '';
        if (pdfUrl) {
            pdfLink = `<a href="${pdfUrl}${pageFragment}" class="view-pdf-btn" target="_blank">View PDF</a>`;
        }

        // Build source link HTML
        let sourceLink = '';
        if (result.URL) {
            sourceLink = `<a href="${escapeHtml(result.URL)}${pageFragment}" target="_blank" class="view-pdf-btn" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">View Source</a>`;
        }

        // Build metadata items
        const metaItems = [];
        if (result.Feed) {
            const feedDisplay = typeof result.Feed === 'object' ? JSON.stringify(result.Feed) : result.Feed;
            metaItems.push(`<div class="meta-item"><span class="meta-label">Feed:</span>${escapeHtml(feedDisplay)}</div>`);
        }
        if (result.PDFPage) {
            metaItems.push(`<div class="meta-item"><span class="meta-label">PDF Page:</span>${result.PDFPage}</div>`);
        }
        if (result.EmpiricalPageNumber) {
            metaItems.push(`<div class="meta-item"><span class="meta-label">Journal Page:</span>${escapeHtml(result.EmpiricalPageNumber)}</div>`);
        }

        // Build JSON toggle (optional)
        let jsonToggle = '';
        let jsonContent = '';
        if (options.showJson && options.index !== undefined) {
            jsonToggle = `<button class="toggle-json" data-index="${options.index}">Show Full JSON</button>`;
            jsonContent = `<div id="taxon-json-${options.index}" class="result-json" style="display: none;"><pre>${JSON.stringify(result, null, 2)}</pre></div>`;
        }

        // Build inline source context viewer for nomenclature (Title)
        const apiUrl = apiBase ? `${apiBase}/api` : '/api';
        let titleContent = '';
        if (result.taxon_id) {
            titleContent = `
                <div data-source-context-viewer
                     data-taxa-id="${escapeHtml(result.taxon_id)}"
                     data-field="nomenclature"
                     data-api-base-url="${apiUrl}"
                     data-original-text="${escapeHtml(result.Title || 'Untitled')}">
                </div>
            `;
        } else {
            titleContent = escapeHtml(result.Title || 'Untitled');
        }

        // Build inline source context viewer for description
        let descriptionContent = '';
        if (result.Description && !options.compact) {
            if (result.taxon_id) {
                descriptionContent = `
                    <div class="result-description" data-source-context-viewer
                         data-taxa-id="${escapeHtml(result.taxon_id)}"
                         data-field="description"
                         data-api-base-url="${apiUrl}"
                         data-original-text="${escapeHtml(result.Description)}">
                    </div>
                `;
            } else {
                const maxLen = options.descriptionLength || 500;
                const truncated = result.Description.length > maxLen;
                descriptionContent = `<div class="result-description">${escapeHtml(result.Description).substring(0, maxLen)}${truncated ? '...' : ''}</div>`;
            }
        }

        card.innerHTML = `
            <div class="result-header">
                <div class="result-title">${titleContent}</div>
                ${similarityPercent !== null ? `<div class="similarity-badge">${similarityPercent}%</div>` : ''}
            </div>
            ${metaItems.length > 0 ? `<div class="result-meta">${metaItems.join('')}</div>` : ''}
            ${descriptionContent}
            <div class="result-actions">
                ${pdfLink}
                ${sourceLink}
                ${jsonToggle}
            </div>
            ${jsonContent}
        `;

        // Add JSON toggle handler if needed
        if (options.showJson && options.index !== undefined) {
            const toggleBtn = card.querySelector('.toggle-json');
            if (toggleBtn) {
                toggleBtn.addEventListener('click', function() {
                    const jsonDiv = document.getElementById(`taxon-json-${options.index}`);
                    if (jsonDiv) {
                        if (jsonDiv.style.display === 'none') {
                            jsonDiv.style.display = 'block';
                            this.textContent = 'Hide Full JSON';
                        } else {
                            jsonDiv.style.display = 'none';
                            this.textContent = 'Show Full JSON';
                        }
                    }
                });
            }
        }

        return card;
    }

    /**
     * Render a taxon result card into a container
     * @param {HTMLElement|string} container - Container element or selector
     * @param {Object} data - Either full result data or reference data with taxon_id
     * @param {Object} options - Rendering options
     * @returns {Promise<HTMLElement>} The rendered card
     */
    async function render(container, data, options = {}) {
        const containerEl = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        if (!containerEl) {
            throw new Error('Container not found');
        }

        // If we have full data (Description present), render directly
        if (data.Description !== undefined) {
            const card = createCard(data, options);
            containerEl.appendChild(card);
            // Trigger initialization of any SourceContextViewer components
            document.dispatchEvent(new CustomEvent('source-context-init'));
            return card;
        }

        // Otherwise, fetch the full taxon data
        if (!data.taxon_id) {
            throw new Error('Either full result data or taxon_id is required');
        }

        // Show loading state
        const loadingEl = document.createElement('div');
        loadingEl.className = 'taxon-result-loading';
        loadingEl.textContent = 'Loading...';
        containerEl.appendChild(loadingEl);

        try {
            const taxonData = await fetchTaxon(data.taxon_id);

            // Merge fetched data with provided data (e.g., similarity from reference)
            const mergedData = { ...taxonData, ...data };

            // Replace loading with card
            const card = createCard(mergedData, {
                ...options,
                similarity: data.similarity // Use similarity from reference if provided
            });
            containerEl.replaceChild(card, loadingEl);
            // Trigger initialization of any SourceContextViewer components
            document.dispatchEvent(new CustomEvent('source-context-init'));
            return card;
        } catch (error) {
            // Show fallback with stored reference data instead of error
            // This handles cases where taxon_id no longer exists (e.g., old searches)
            if (data.title || data.similarity !== undefined) {
                // We have enough data from the stored reference to show something useful
                const fallbackData = {
                    Title: data.title || 'Unknown Taxon',
                    Similarity: data.similarity,
                    Description: '(Full details unavailable - taxon record may have been updated)',
                };
                const card = createCard(fallbackData, { ...options, compact: true });
                card.classList.add('taxon-result-fallback');
                containerEl.replaceChild(card, loadingEl);
                // Trigger initialization of any SourceContextViewer components
                document.dispatchEvent(new CustomEvent('source-context-init'));
                return card;
            }

            // No fallback data available, show error
            loadingEl.className = 'taxon-result-error';
            loadingEl.innerHTML = `
                <div class="error-title">${escapeHtml(data.title || data.taxon_id)}</div>
                <div class="error-message">Failed to load: ${escapeHtml(error.message)}</div>
            `;
            return loadingEl;
        }
    }

    /**
     * Render multiple taxon results into a container
     * @param {HTMLElement|string} container - Container element or selector
     * @param {Array} results - Array of result data or references
     * @param {Object} options - Rendering options
     * @returns {Promise<HTMLElement[]>} Array of rendered cards
     */
    async function renderAll(container, results, options = {}) {
        const containerEl = typeof container === 'string'
            ? document.querySelector(container)
            : container;

        if (!containerEl) {
            throw new Error('Container not found');
        }

        // Clear container if requested
        if (options.clear) {
            containerEl.innerHTML = '';
        }

        // Render all results
        const cards = await Promise.all(
            results.map((data, index) => render(containerEl, data, {
                ...options,
                index: options.showJson ? index : undefined
            }))
        );

        return cards;
    }

    /**
     * Clear the taxon cache
     */
    function clearCache() {
        taxonCache.clear();
    }

    // Public API
    return {
        init,
        fetchTaxon,
        createCard,
        render,
        renderAll,
        clearCache,
        escapeHtml,
        buildPdfUrl
    };
})();

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TaxonResultWidget;
}
