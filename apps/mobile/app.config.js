const baseUrl = process.env.EXPO_PUBLIC_BASE_URL || '/';

module.exports = {
  expo: {
    name: 'GuideLens',
    slug: 'guidelens',
    version: '0.1.0',
    orientation: 'portrait',
    scheme: 'guidelens',
    userInterfaceStyle: 'dark',
    icon: './assets/icon.png',
    ios: {
      supportsTablet: true,
      bundleIdentifier: 'edu.sfbu.guidelens',
      infoPlist: {
        NSCameraUsageDescription: 'GuideLens uses the camera to describe nearby objects and obstacles.',
        NSMicrophoneUsageDescription: 'GuideLens can accept voice controls.',
      },
    },
    android: {
      package: 'edu.sfbu.guidelens',
      permissions: ['CAMERA', 'RECORD_AUDIO'],
    },
    web: {
      bundler: 'metro',
      output: 'single',
      name: 'GuideLens — AI vision companion',
      favicon: './assets/favicon.png',
    },
    plugins: [
      'expo-router',
      'expo-asset',
      'onnxruntime-react-native',
      ['expo-camera', { cameraPermission: 'Allow GuideLens to identify nearby objects and obstacles.' }],
    ],
    experiments: {
      typedRoutes: true,
      baseUrl,
    },
  },
};
