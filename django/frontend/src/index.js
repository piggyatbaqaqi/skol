import React from 'react';
import { createRoot } from 'react-dom/client';
import PDFViewer from './PDFViewer';
import './styles.css';

// Get configuration from the page
const container = document.getElementById('pdf-viewer-root');

if (container) {
  const pdfUrl = container.dataset.pdfUrl;
  const initialPage = parseInt(container.dataset.initialPage || '1', 10);
  const title = container.dataset.title || 'PDF Viewer';

  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <PDFViewer
        pdfUrl={pdfUrl}
        initialPage={initialPage}
        title={title}
      />
    </React.StrictMode>
  );
}
