import React from 'react';
import { createRoot } from 'react-dom/client';
import CodeBlockExample from './CodeBlockExample';

// Mount CodeBlockExample to the DOM
const container = document.getElementById('example-code-block');
if (container) {
  const root = createRoot(container);
  const text = container.getAttribute('data-text') || 'campanulate pileus, stipe stains blue when cut';
  const language = container.getAttribute('data-language') || 'text';
  const showLineNumbers = container.getAttribute('data-show-line-numbers') === 'true';

  root.render(
    <CodeBlockExample
      text={text}
      language={language}
      showLineNumbers={showLineNumbers}
    />
  );
}
