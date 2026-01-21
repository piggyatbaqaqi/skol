import { useState, useCallback, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
// CSS files from react-pdf source (4 levels up from src to reach parent of skol)
import '../../../../react-pdf/packages/react-pdf/src/Page/AnnotationLayer.css';
import '../../../../react-pdf/packages/react-pdf/src/Page/TextLayer.css';

// Set the worker source - this file is copied by webpack
pdfjs.GlobalWorkerOptions.workerSrc = `${window.API_BASE}/static/js/pdf.worker.min.mjs`;

const PDFViewer = ({ pdfUrl, initialPage = 1, title = 'PDF Viewer' }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(initialPage);
  const [scale, setScale] = useState(1.0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Update page number when initialPage changes
  useEffect(() => {
    if (initialPage && initialPage > 0) {
      setPageNumber(initialPage);
    }
  }, [initialPage]);

  const onDocumentLoadSuccess = useCallback(({ numPages }) => {
    setNumPages(numPages);
    setLoading(false);
    // Ensure we don't exceed the number of pages
    if (pageNumber > numPages) {
      setPageNumber(numPages);
    }
  }, [pageNumber]);

  const onDocumentLoadError = useCallback((error) => {
    console.error('PDF load error:', error);
    setError(error.message || 'Failed to load PDF');
    setLoading(false);
  }, []);

  // Handle navigation from URL hash or internal links
  const onItemClick = useCallback(({ pageNumber: targetPage }) => {
    if (targetPage >= 1 && targetPage <= (numPages || Infinity)) {
      setPageNumber(targetPage);
    }
  }, [numPages]);

  const goToPrevPage = () => {
    setPageNumber((prev) => Math.max(prev - 1, 1));
  };

  const goToNextPage = () => {
    setPageNumber((prev) => Math.min(prev + 1, numPages || prev));
  };

  const goToPage = (e) => {
    const page = parseInt(e.target.value, 10);
    if (page >= 1 && page <= numPages) {
      setPageNumber(page);
    }
  };

  const zoomIn = () => {
    setScale((prev) => Math.min(prev + 0.25, 3.0));
  };

  const zoomOut = () => {
    setScale((prev) => Math.max(prev - 0.25, 0.5));
  };

  const resetZoom = () => {
    setScale(1.0);
  };

  if (error) {
    return (
      <div className="pdf-error">
        <h2>Error Loading PDF</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="pdf-viewer-container">
      <div className="pdf-header">
        <h2>{title}</h2>
      </div>

      <div className="pdf-toolbar">
        <div className="toolbar-section">
          <button onClick={goToPrevPage} disabled={pageNumber <= 1}>
            Previous
          </button>
          <span className="page-info">
            Page{' '}
            <input
              type="number"
              value={pageNumber}
              onChange={goToPage}
              min={1}
              max={numPages || 1}
              className="page-input"
            />{' '}
            of {numPages || '...'}
          </span>
          <button onClick={goToNextPage} disabled={pageNumber >= numPages}>
            Next
          </button>
        </div>

        <div className="toolbar-section">
          <button onClick={zoomOut} disabled={scale <= 0.5}>
            -
          </button>
          <span className="zoom-info">{Math.round(scale * 100)}%</span>
          <button onClick={zoomIn} disabled={scale >= 3.0}>
            +
          </button>
          <button onClick={resetZoom}>Reset</button>
        </div>

        <div className="toolbar-section">
          <a
            href={`${pdfUrl}?download=true`}
            className="download-btn"
            download
          >
            Download
          </a>
        </div>
      </div>

      <div className="pdf-content">
        {loading && <div className="pdf-loading">Loading PDF...</div>}

        <Document
          file={pdfUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={onDocumentLoadError}
          onItemClick={onItemClick}
          loading={<div className="pdf-loading">Loading document...</div>}
          enableUrlHash={true}
          syncUrlHash={true}
        >
          <Page
            pageNumber={pageNumber}
            scale={scale}
            loading={<div className="pdf-loading">Loading page...</div>}
            renderTextLayer={true}
            renderAnnotationLayer={true}
          />
        </Document>
      </div>
    </div>
  );
};

export default PDFViewer;
