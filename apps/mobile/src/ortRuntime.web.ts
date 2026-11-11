/**
 * One ONNX Runtime Web instance for every on-device model in the browser.
 *
 * Both bundles install themselves as the global `ort`, so whichever model
 * loaded first used to decide the build for all of them. The WebGPU bundle
 * also contains the CPU (wasm) backend, so it is loaded whenever the browser
 * exposes WebGPU, and each model picks its own execution provider.
 */
export type OrtTensor = { data: Float32Array };
export type OrtSession = { run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>> };
export type OrtRuntime = {
  env: { wasm: { wasmPaths: string; numThreads: number } };
  InferenceSession: { create: (url: string, options: object) => Promise<OrtSession> };
  Tensor: new (type: string, data: Float32Array, dimensions: number[]) => unknown;
};

type OrtGlobal = typeof globalThis & { ort?: OrtRuntime; __ortHasWebGpu?: boolean };

export function assetUrl(path: string) {
  const base = document.querySelector('script[src*="/_expo/"]')?.getAttribute('src')?.split('/_expo/')[0] || '';
  return `${base}/${path}`.replace(/([^:]\/)\/+/, '$1');
}

export function browserHasWebGpu() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
}

let runtimePromise: Promise<OrtRuntime> | null = null;

export function getOrtRuntime(): Promise<OrtRuntime> {
  if (!runtimePromise) {
    runtimePromise = new Promise((resolve, reject) => {
      const scope = globalThis as OrtGlobal;
      const configure = (ort: OrtRuntime) => {
        ort.env.wasm.wasmPaths = assetUrl('ort/');
        ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
        resolve(ort);
      };
      if (scope.ort) return configure(scope.ort);
      const webgpu = browserHasWebGpu();
      const script = document.createElement('script');
      script.src = assetUrl(webgpu ? 'ort/ort.webgpu.min.js' : 'ort/ort.wasm.min.js');
      script.onload = () => {
        scope.__ortHasWebGpu = webgpu;
        if (scope.ort) configure(scope.ort);
        else reject(new Error('ONNX Runtime did not initialise'));
      };
      script.onerror = () => reject(new Error('ONNX Runtime could not be loaded'));
      document.head.appendChild(script);
    });
  }
  return runtimePromise;
}

/** True when the loaded runtime includes the WebGPU execution provider. */
export function runtimeHasWebGpu() {
  return Boolean((globalThis as OrtGlobal).__ortHasWebGpu);
}
