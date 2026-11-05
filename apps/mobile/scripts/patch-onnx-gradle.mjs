import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const gradleFile = join(
  scriptDirectory,
  '..',
  '..',
  '..',
  'node_modules',
  'onnxruntime-react-native',
  'android',
  'build.gradle',
);

const incompatibleCheck =
  'if (VersionNumber.parse(REACT_NATIVE_VERSION) < VersionNumber.parse("0.71")) {';
const compatibleCheck =
  'if (false) { // GuideLens uses React Native 0.71+; Gradle 9 removed VersionNumber.';

const source = await readFile(gradleFile, 'utf8');
const occurrences = source.split(incompatibleCheck).length - 1;

if (occurrences !== 1) {
  throw new Error(
    `Expected exactly one ONNX Runtime VersionNumber check, found ${occurrences}.`,
  );
}

await writeFile(gradleFile, source.replace(incompatibleCheck, compatibleCheck));
