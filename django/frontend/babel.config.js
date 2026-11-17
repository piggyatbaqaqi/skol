// Shared by webpack (babel-loader) and Jest (babel-jest), so the bundle
// and the tests compile JSX the same way.  Under Jest (NODE_ENV=test)
// preset-env targets the running Node instead of browsers.
module.exports = (api) => {
  const test = api.env('test');
  return {
    presets: [
      ['@babel/preset-env', test ? { targets: { node: 'current' } } : {}],
      ['@babel/preset-react', { runtime: 'automatic' }],
    ],
  };
};
