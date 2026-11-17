// Jest setup: what the webpack bundle provides that jsdom does not.
import '@testing-library/jest-dom';
import React from 'react';

// webpack.config.js's ProvidePlugin makes React a global for code that
// uses React.xyz without importing it.
global.React = React;
