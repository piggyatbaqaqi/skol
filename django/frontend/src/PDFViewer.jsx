import { useRef } from 'react';
import { Viewer, Worker } from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';

// Import styles
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';

// PDF.js worker version must match the pdfjs-dist version
const PDFJS_VERSION = '3.11.174';

const PDFViewer = ({ pdfUrl, initialPage = 1, title = 'PDF Viewer' }) => {
  const containerRef = useRef(null);

  // Create the default layout plugin with search enabled
  // The default layout includes: toolbar, sidebar (thumbnails, bookmarks, attachments), and search
  const defaultLayoutPluginInstance = defaultLayoutPlugin({
    sidebarTabs: (defaultTabs) => defaultTabs,
    toolbarPlugin: {
      fullScreenPlugin: {
        // Full screen button in toolbar
      },
      searchPlugin: {
        // Search is enabled by default in the toolbar
      },
    },
  });

  return (
    <div className="pdf-viewer-container" ref={containerRef}>
      <div className="pdf-header">
        <h2>{title}</h2>
        <a
          href={`${pdfUrl}?download=true`}
          className="download-btn"
          download
        >
          Download PDF
        </a>
      </div>

      <div className="pdf-viewer-wrapper">
        <Worker workerUrl={`https://unpkg.com/pdfjs-dist@${PDFJS_VERSION}/build/pdf.worker.min.js`}>
          <Viewer
            fileUrl={pdfUrl}
            plugins={[defaultLayoutPluginInstance]}
            initialPage={initialPage - 1}
            defaultScale={1.0}
          />
        </Worker>
      </div>
    </div>
  );
};

export default PDFViewer;
