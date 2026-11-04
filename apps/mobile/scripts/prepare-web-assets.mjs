import { cp, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const mobile = join(here, '..');
const publicDir = join(mobile, 'public');

await mkdir(join(publicDir, 'models'), { recursive: true });
await cp(join(mobile, 'assets', 'models', 'yolox_nano.onnx'), join(publicDir, 'models', 'yolox_nano.onnx'));
