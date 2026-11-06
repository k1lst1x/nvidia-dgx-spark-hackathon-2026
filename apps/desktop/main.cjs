const { app, BrowserWindow, session } = require('electron');
const { createServer } = require('node:http');
const { readFile, stat } = require('node:fs/promises');
const { extname, join, normalize } = require('node:path');

const root = join(__dirname, 'web');
if (process.env.GUIDELENS_DEVTOOLS === '1') app.commandLine.appendSwitch('remote-debugging-port', '9222');
const mime = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.onnx': 'application/octet-stream', '.png': 'image/png', '.svg': 'image/svg+xml', '.wasm': 'application/wasm',
  '.ico': 'image/x-icon',
};

async function serve(request, response) {
  const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
  const candidate = normalize(join(root, pathname));
  let file = candidate.startsWith(root) ? candidate : join(root, 'index.html');
  try {
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
  } catch {
    file = join(root, 'index.html');
  }
  try {
    const body = await readFile(file);
    response.writeHead(200, {
      'Content-Type': mime[extname(file)] || 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Access-Control-Allow-Origin': '*',
      'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; font-src 'self' http://assets; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' https: http:; worker-src 'self' blob: https://cdn.jsdelivr.net",
    });
    response.end(body);
  } catch {
    response.writeHead(404).end('Not found');
  }
}

app.whenReady().then(() => {
  const server = createServer((request, response) => void serve(request, response));
  server.listen(0, '127.0.0.1', () => {
    const address = server.address();
    const origin = `http://127.0.0.1:${address.port}`;
    const isTrusted = (url) => url.startsWith(`${origin}/`) || url === origin;
    session.defaultSession.webRequest.onBeforeRequest({ urls: ['http://assets/*'] }, (details, callback) => {
      callback({ redirectURL: `${origin}/assets${new URL(details.url).pathname}` });
    });
    session.defaultSession.setPermissionCheckHandler((contents, permission) => permission === 'media' && isTrusted(contents.getURL()));
    session.defaultSession.setPermissionRequestHandler((contents, permission, callback) => callback(permission === 'media' && isTrusted(contents.getURL())));
    const appIcon = join(__dirname, 'assets', 'icon.png');
    if (process.platform === 'darwin' && app.dock) {
      try { app.dock.setIcon(appIcon); } catch {}
    }
    const window = new BrowserWindow({
      width: 1180,
      height: 820,
      minWidth: 390,
      minHeight: 680,
      backgroundColor: '#07110F',
      title: 'GuideLens',
      icon: appIcon,
      webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
    });
    window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    window.webContents.on('console-message', (event) => {
      console.log(`[renderer:${event.level}] ${event.message}`);
    });
    window.webContents.on('will-navigate', (event, url) => {
      if (!isTrusted(url)) event.preventDefault();
    });
    window.loadURL(`${origin}${process.env.GUIDELENS_START_PATH || ''}`);
    if (process.env.GUIDELENS_DEVTOOLS === '1') window.webContents.openDevTools({ mode: 'detach' });
  });
  app.on('before-quit', () => server.close());
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
