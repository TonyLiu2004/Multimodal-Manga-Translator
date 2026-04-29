const path = require('path');
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
