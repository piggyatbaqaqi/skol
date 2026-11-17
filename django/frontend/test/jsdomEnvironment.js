// jest-environment-jsdom, with the optional native `canvas` package hidden.
//
// pdfjs-dist pulls in canvas@2.11.2 as an optional dependency, and its
// prebuilt binary dies with SIGBUS under Node 22.  jsdom loads canvas
// whenever it is installed, and treats MODULE_NOT_FOUND as "no canvas", so
// reporting it missing lets the widget tests run however `npm install` was
// invoked.  (build-deb.sh installs with --omit=optional and never has it.)
const Module = require('module');

const resolve = Module._resolveFilename;
Module._resolveFilename = function resolveWithoutCanvas(request, ...rest) {
  if (request === 'canvas' || request.startsWith('canvas/')) {
    const err = new Error(`Cannot find module '${request}' (hidden from Jest)`);
    err.code = 'MODULE_NOT_FOUND';
    throw err;
  }
  return resolve.call(this, request, ...rest);
};

module.exports = require('jest-environment-jsdom').TestEnvironment;
