import { cp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const desktop = join(dirname(fileURLToPath(import.meta.url)), '..');
const web = join(desktop, 'web');
await rm(web, { recursive: true, force: true });
await cp(join(desktop, '..', 'mobile', 'dist'), web, { recursive: true });
