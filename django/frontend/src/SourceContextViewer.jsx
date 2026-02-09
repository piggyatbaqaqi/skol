/**
 * SourceContextViewer - Inline toggle between text and source context
 *
 * Displays either the original text or a windowed view into the source with:
 * - All spans highlighted in a single window
 * - Toggle button to switch between text and context views
 */
import React, { useState, useCallback, useRef, useEffect } from 'react';
import './SourceContextViewer.css';

/**
 * Read the Django CSRF token from the cookie.
 */
function getCSRFToken() {
  const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
  return match ? match[1] : '';
}

/**
 * SourceContextViewer Component
 *
 * @param {Object} props
 * @param {string} props.taxaId - Taxa document ID
 * @param {string} props.field - 'nomenclature' or 'description'
 * @param {string} props.originalText - The original text to display (optional)
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {number} props.contextChars - Characters of context (default: 500)
 * @param {string} props.taxaDb - Database name (default: 'skol_taxa_dev')
 */
const SourceContextViewer = ({
  taxaId,
  field = 'description',
  originalText = '',
  apiBaseUrl = '/api',
  contextChars = 500,
  taxaDb = 'skol_taxa_dev',
}) => {
  const [showContext, setShowContext] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [context, setContext] = useState(null);
  const contentRef = useRef(null);

  /**
   * Fetch context from the API
   */
  const fetchContext = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        field,
        context_chars: contextChars.toString(),
        taxa_db: taxaDb,
      });

      const response = await fetch(
        `${apiBaseUrl}/taxa/${taxaId}/context/?${params}`,
        {
          method: 'GET',
          headers: {
            'X-CSRFToken': getCSRFToken(),
          },
          credentials: 'same-origin',
        }
      );

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setContext(data);
    } catch (err) {
      console.error('Failed to fetch source context:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, taxaId, field, contextChars, taxaDb]);

  /**
   * Toggle between text and context view
   */
  const handleToggle = useCallback(() => {
    if (!showContext && !context && !loading) {
      fetchContext();
    }
    setShowContext((prev) => !prev);
  }, [showContext, context, loading, fetchContext]);

  /**
   * Scroll to first highlight when content loads
   */
  useEffect(() => {
    if (showContext && context && contentRef.current) {
      const mark = contentRef.current.querySelector('mark');
      if (mark) {
        mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [showContext, context]);

  return (
    <div className="source-context-viewer source-context-inline">
      {/* Header with label and toggle button */}
      <div className="source-context-header">
        <button
          type="button"
          className="source-context-toggle-btn"
          onClick={handleToggle}
          aria-expanded={showContext}
          title={showContext ? 'Show text' : 'Show source context'}
        >
          {showContext ? 'Text' : 'Context'}
        </button>
      </div>

      {/* Content area - either original text or context viewer */}
      {!showContext ? (
        /* Original text view */
        <div className="source-context-original-text">
          {originalText}
        </div>
      ) : (
        /* Context viewer */
        <div
          className="source-context-panel"
          role="region"
          aria-label="Source context"
        >
          {loading && (
            <div className="source-context-loading" role="status" aria-live="polite">
              <div className="source-context-skeleton" />
              <div className="source-context-skeleton source-context-skeleton-short" />
              Loading...
            </div>
          )}

          {error && (
            <div className="source-context-error" role="alert">
              {error}
            </div>
          )}

          {!loading && !error && context && (
            <>
              {/* Metadata bar */}
              <div className="source-context-meta">
                {context.pdf_page && (
                  <span className="source-context-page">
                    PDF p.{context.pdf_page}
                    {context.pdf_label && context.pdf_label !== String(context.pdf_page) && (
                      <> ({context.pdf_label})</>
                    )}
                  </span>
                )}
                {context.empirical_page && (
                  <span className="source-context-page">
                    J.p.{context.empirical_page}
                  </span>
                )}
                {context.total_spans > 1 && (
                  <span className="source-context-span-info">
                    {context.total_spans} spans
                  </span>
                )}
              </div>

              {/* Source text content with all spans highlighted */}
              <div
                ref={contentRef}
                className="source-context-content"
                dangerouslySetInnerHTML={{ __html: context.source_text }}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default SourceContextViewer;
