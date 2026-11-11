// Loads a TypeScript source file for a Node test, with optional stubbed imports.
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');

module.exports = function loadTs(relativePath, stubs = {}) {
  const file = path.join(__dirname, '..', relativePath);
  const output = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.React, esModuleInterop: true },
  }).outputText;
  const loaded = new Module(file);
  loaded.filename = file;
  loaded.paths = Module._nodeModulePaths(path.dirname(file));
  const realRequire = loaded.require.bind(loaded);
  loaded.require = (id) => (id in stubs ? stubs[id] : realRequire(id));
  loaded._compile(output, file);
  return loaded.exports;
};
