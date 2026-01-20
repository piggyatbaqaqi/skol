const path = require('path');
const CopyWebpackPlugin = require('copy-webpack-plugin');

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
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react'],
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
  },
  plugins: [
    new CopyWebpackPlugin({
      patterns: [
        {
          // Copy the pdf.js worker from react-pdf's pdfjs-dist dependency
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
