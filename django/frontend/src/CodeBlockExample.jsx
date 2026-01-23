import React from 'react';
import { CopyBlock, github } from 'react-code-blocks';

/**
 * CodeBlockExample component - displays example search query with copy functionality
 */
const CodeBlockExample = ({ text, language = 'text', showLineNumbers = false }) => {
  // Custom theme based on github but with adjusted colors for better visibility
  const customTheme = {
    ...github,
    backgroundColor: '#f9f9f9',
    textColor: '#333',
    buttonBackgroundColor: '#667eea',
    buttonTextColor: '#fff',
  };

  return (
    <div
      style={{
        marginBottom: '20px',
        marginTop: '10px',
        borderRadius: '6px',
        overflow: 'hidden',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
        border: '2px solid #e0e0e0',
        backgroundColor: '#f9f9f9'
      }}
      className="code-block-example"
    >
      <style>{`
        .code-block-example button {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
          color: white !important;
          border: none !important;
          padding: 6px 12px !important;
          border-radius: 4px !important;
          font-size: 12px !important;
          font-weight: 600 !important;
          cursor: pointer !important;
          transition: transform 0.2s !important;
        }
        .code-block-example button:hover {
          transform: translateY(-2px) !important;
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        }
      `}</style>
      <CopyBlock
        text={text}
        language={language}
        showLineNumbers={showLineNumbers}
        theme={customTheme}
        wrapLines={true}
        codeBlock
        customStyle={{
          fontSize: '14px',
          padding: '16px',
          fontFamily: "'Courier New', monospace"
        }}
      />
    </div>
  );
};

export default CodeBlockExample;
