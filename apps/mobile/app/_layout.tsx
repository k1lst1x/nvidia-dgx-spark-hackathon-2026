import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { LogBox } from 'react-native';

const ignoredTfliteNotice = 'Created TensorFlow Lite';
LogBox.ignoreLogs([ignoredTfliteNotice, `INFO: ${ignoredTfliteNotice}`]);

// On web, the dev client mirrors `console.info` messages in an intrusive
// bottom-left overlay. The TFLite runtime emits this startup notice every time
// its model initializes; filter only that known non-actionable message.
const suppressTfliteStartupNotice = (method: (...args: unknown[]) => void) => (...args: unknown[]) => {
  if (args.some((value) => String(value).includes(ignoredTfliteNotice))) return;
  method(...args);
};
console.info = suppressTfliteStartupNotice(console.info);
console.log = suppressTfliteStartupNotice(console.log);
console.warn = suppressTfliteStartupNotice(console.warn);

export default function RootLayout(){return <><StatusBar style="light"/><Stack screenOptions={{headerShown:false,contentStyle:{backgroundColor:'#07110F'}}}/></>}
