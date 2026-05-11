const path = require('path');
const fs = require('fs');
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
const targetSubpath = 'expo-asset/build/resolveAssetSource';
const targetFile = path.join(
  __dirname,
  'node_modules',
  'expo-asset',
  'build',
  'resolveAssetSource.js'
);

const ioniconsFont = path.join(__dirname, 'assets', 'fonts', 'Ionicons.ttf');
const ioniconsSourceFont = path.join(
  __dirname,
  'node_modules',
  '@expo',
  'vector-icons',
  'build',
  'vendor',
  'react-native-vector-icons',
  'Fonts',
  'Ionicons.ttf'
);

// The deployed build needs Ionicons outside assets/node_modules, but local Metro
// also needs the generated file to exist before it computes SHA-1 hashes.
if (!fs.existsSync(ioniconsFont) && fs.existsSync(ioniconsSourceFont)) {
  fs.mkdirSync(path.dirname(ioniconsFont), { recursive: true });
  fs.copyFileSync(ioniconsSourceFont, ioniconsFont);
}

const defaultResolveRequest = config.resolver.resolveRequest;

config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (moduleName === targetSubpath) {
    return {
      type: 'sourceFile',
      filePath: targetFile,
    };
  }

  const normalized =
    typeof moduleName === 'string' ? moduleName.replace(/\\/g, '/') : '';
  if (normalized.endsWith('Fonts/Ionicons.ttf')) {
    return {
      type: 'sourceFile',
      filePath: ioniconsFont,
    };
  }

  if (defaultResolveRequest) {
    return defaultResolveRequest(context, moduleName, platform);
  }
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;
