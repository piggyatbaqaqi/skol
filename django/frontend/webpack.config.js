const path = require('path');
const webpack = require('webpack');
const CopyWebpackPlugin = require('copy-webpack-plugin');

// Path to local react-pdf (neighbor directory to skol)
const reactPdfPath = path.resolve(__dirname, '../../../react-pdf/packages/react-pdf');

module.exports = {
  entry: './src/index.js',
  output: {
    filename: 'pdf-viewer.bundle.js',
    path: path.resolve(__dirname, '../static/js'),
    clean: false, // Don't clean the directory (other files may exist)
  },
  module: {
    rules: [
      {
        // Compile TypeScript/JSX from both local src and react-pdf source
        test: /\.(js|jsx|ts|tsx)$/,
        include: [
          path.resolve(__dirname, 'src'),
          path.join(reactPdfPath, 'src'),
        ],
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              '@babel/preset-env',
              ['@babel/preset-react', { runtime: 'automatic' }],
              '@babel/preset-typescript',
            ],
          },
        },
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  resolve: {
    // Map .js imports to .ts/.tsx files (ESM convention used in react-pdf source)
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
    extensionAlias: {
      '.js': ['.ts', '.tsx', '.js'],
    },
    // Resolve all node_modules from frontend directory (for react-pdf source compilation)
    modules: [path.resolve(__dirname, 'node_modules'), 'node_modules'],
    alias: {
      // Point to the source of local react-pdf (compile from TypeScript source)
      'react-pdf': path.join(reactPdfPath, 'src'),
    },
  },
  plugins: [
    // Provide React globally for code that uses React.xyz without importing
    new webpack.ProvidePlugin({
      React: 'react',
    }),
    new CopyWebpackPlugin({
      patterns: [
        {
          // Copy the pdf.js worker from pdfjs-dist (via react-pdf's node_modules or local)
          from: 'node_modules/pdfjs-dist/build/pdf.worker.min.mjs',
          to: path.resolve(__dirname, '../static/js/pdf.worker.min.mjs'),
        },
      ],
    }),
  ],
  // Optimize for production
  optimization: {
    minimize: true,
  },
};
