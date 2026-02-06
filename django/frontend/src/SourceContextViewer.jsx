/**
 * SourceContextViewer - Expandable viewer for source text context
 *
 * Displays a windowed view into the source article.txt with:
 * - Highlighted nomenclature or description text
 * - Gap indicators for coalesced content
 * - Navigation between multiple spans
 * - Scroll navigation
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
 * @param {string} props.apiBaseUrl - Base URL for API calls (default: '/api')
 * @param {number} props.contextChars - Characters of context (default: 500)
 * @param {string} props.taxaDb - Database name (default: 'skol_taxa_dev')
 */
const SourceContextViewer = ({
  taxaId,
  field = 'description',
  apiBaseUrl = '/api',
  contextChars = 500,
  taxaDb = 'skol_taxa_dev',
}) => {
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [context, setContext] = useState(null);
  const [spanIndex, setSpanIndex] = useState(0);
  const contentRef = useRef(null);

  /**
   * Fetch context from the API
   */
  const fetchContext = useCallback(async (index) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        field,
        span_index: index.toString(),
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
      setSpanIndex(data.span_index);
    } catch (err) {
      console.error('Failed to fetch source context:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, taxaId, field, contextChars, taxaDb]);

  /**
   * Toggle expanded state
   */
  const handleToggle = useCallback(() => {
    if (!expanded && !context && !loading) {
      fetchContext(0);
    }
    setExpanded((prev) => !prev);
  }, [expanded, context, loading, fetchContext]);

  /**
   * Navigate to previous span
   */
  const handlePrevSpan = useCallback(() => {
    if (context && context.prev_span_index !== null) {
      fetchContext(context.prev_span_index);
    }
  }, [context, fetchContext]);

  /**
   * Navigate to next span
   */
  const handleNextSpan = useCallback(() => {
    if (context && context.next_span_index !== null) {
      fetchContext(context.next_span_index);
    }
  }, [context, fetchContext]);

  /**
   * Scroll to highlight when content loads
   */
  useEffect(() => {
    if (expanded && context && contentRef.current) {
      const mark = contentRef.current.querySelector('mark');
      if (mark) {
        mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [expanded, context]);

  const fieldLabel = field === 'nomenclature' ? 'Nomenclature' : 'Description';

  return (
    <div className="source-context-viewer">
      <button
        type="button"
        className="source-context-toggle"
        onClick={handleToggle}
        aria-expanded={expanded}
        aria-controls={`context-panel-${taxaId}-${field}`}
      >
        {expanded ? `Hide ${fieldLabel} Context` : `Show ${fieldLabel} Context`}
      </button>

      {expanded && (
        <div
          id={`context-panel-${taxaId}-${field}`}
          className="source-context-panel"
          role="region"
          aria-label={`${fieldLabel} source context`}
        >
          {loading && (
            <div className="source-context-loading" role="status" aria-live="polite">
              <div className="source-context-skeleton" />
              <div className="source-context-skeleton" />
              <div className="source-context-skeleton source-context-skeleton-short" />
              Loading context...
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
                    PDF Page {context.pdf_page}
                    {context.pdf_label && context.pdf_label !== String(context.pdf_page) && (
                      <> (Label: {context.pdf_label})</>
                    )}
                  </span>
                )}
                {context.empirical_page && (
                  <span className="source-context-page">
                    Journal Page {context.empirical_page}
                  </span>
                )}
                {context.total_spans > 1 && (
                  <span className="source-context-span-info">
                    Span {spanIndex + 1} of {context.total_spans}
                  </span>
                )}
              </div>

              {/* Gap indicator before */}
              {context.has_gap_before && (
                <div className="source-context-gap" role="status">
                  Gap: {context.gap_size_before} characters omitted
                </div>
              )}

              {/* Source text content */}
              <div
                ref={contentRef}
                className="source-context-content"
                dangerouslySetInnerHTML={{ __html: context.source_text }}
              />

              {/* Gap indicator after */}
              {context.has_gap_after && (
                <div className="source-context-gap" role="status">
                  Gap: {context.gap_size_after} characters omitted
                </div>
              )}

              {/* Navigation buttons */}
              {context.total_spans > 1 && (
                <div className="source-context-nav">
                  <button
                    type="button"
                    className="source-context-nav-btn"
                    onClick={handlePrevSpan}
                    disabled={context.prev_span_index === null}
                    aria-label="Previous span"
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    className="source-context-nav-btn"
                    onClick={handleNextSpan}
                    disabled={context.next_span_index === null}
                    aria-label="Next span"
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default SourceContextViewer;
