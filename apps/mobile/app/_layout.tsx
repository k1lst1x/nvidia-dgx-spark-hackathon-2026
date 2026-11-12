import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { LogBox } from 'react-native';

const ignoredTfliteNotice = 'Created TensorFlow Lite';
LogBox.ignoreLogs([
  ignoredTfliteNotice,
  `INFO: ${ignoredTfliteNotice}`,
  'XNNPACK delegate for CPU',
  /Created TensorFlow Lite/i,
  /INFO:\s*Created TensorFlow Lite/i,
  /XNNPACK delegate/i,
]);

// The TFLite / XNNPACK runtime emits startup notices (e.g. "INFO: Created TensorFlow Lite XNNPACK delegate for CPU")
// via stderr / console.error every time the model initializes.
// On mobile devices and web, these trigger intrusive error banners / LogBox toasts.
// Suppress these non-actionable startup logs across all console levels.
const isTfliteNotice = (value: unknown): boolean => {
  if (value == null) return false;
  const str =
    typeof value === 'string'
      ? value
      : String((value as { message?: unknown })?.message || (value as { stack?: unknown })?.stack || value);
  return (
    str.includes(ignoredTfliteNotice) ||
    str.includes('XNNPACK delegate') ||
    /Created TensorFlow Lite/i.test(str)
  );
};

const suppressTfliteStartupNotice = (method: (...args: unknown[]) => void) => (...args: unknown[]) => {
  if (args.some(isTfliteNotice) || isTfliteNotice(args.map((v) => String(v)).join(' '))) return;
  method(...args);
};

console.error = suppressTfliteStartupNotice(console.error);
console.warn = suppressTfliteStartupNotice(console.warn);
console.info = suppressTfliteStartupNotice(console.info);
console.log = suppressTfliteStartupNotice(console.log);

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: '#07110F' } }} />
    </>
  );
}
