import { cp, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = dirname(fileURLToPath(import.meta.url));
const mobile = join(here, '..');
const publicDir = join(mobile, 'public');
const require = createRequire(import.meta.url);
const ortDist = dirname(require.resolve('onnxruntime-web'));

await mkdir(join(publicDir, 'models'), { recursive: true });
await mkdir(join(publicDir, 'ort'), { recursive: true });
await cp(join(mobile, 'assets', 'models', 'yolox_nano.onnx'), join(publicDir, 'models', 'yolox_nano.onnx'));
for (const name of ['ort.wasm.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await cp(join(ortDist, name), join(publicDir, 'ort', name));
}

const faviconFiles = [
  'favicon.ico',
  'favicon.svg',
  'favicon.png',
  'favicon-32x32.png',
  'favicon-16x16.png',
  'apple-touch-icon.png',
];

for (const name of faviconFiles) {
  await cp(join(mobile, 'assets', name), join(publicDir, name));
}

await cp(
  join(mobile, 'assets', 'index.template.html'),
  join(publicDir, 'index.html'),
);

