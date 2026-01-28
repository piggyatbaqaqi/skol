const path = require('path');
const webpack = require('webpack');

module.exports = {
  entry: {
    'pdf-viewer': './src/index.js',
    'code-block': './src/codeBlockEntry.js',
    'vocab-tree': './src/vocabTreeEntry.js',
    'fungarium-select': './src/fungariumSelectEntry.js',
    'identifier-type-select': './src/identifierTypeSelectEntry.js',
    'collection-select': './src/collectionSelectEntry.js',
  },
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, '../static/js'),
    clean: false, // Don't clean the directory (other files may exist)
  },
  module: {
    rules: [
      {
        // Compile JSX from local src
        test: /\.(js|jsx)$/,
        include: [
          path.resolve(__dirname, 'src'),
        ],
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              '@babel/preset-env',
              ['@babel/preset-react', { runtime: 'automatic' }],
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
    extensions: ['.js', '.jsx'],
    modules: [path.resolve(__dirname, 'node_modules'), 'node_modules'],
  },
  plugins: [
    // Provide React globally for code that uses React.xyz without importing
    new webpack.ProvidePlugin({
      React: 'react',
    }),
  ],
  // Optimize for production
  optimization: {
    minimize: true,
  },
};
