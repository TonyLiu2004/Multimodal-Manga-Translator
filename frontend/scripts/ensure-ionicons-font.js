/**
 * Copies Ionicons.ttf into assets/fonts/ so Metro can emit /assets/fonts/... URLs
 * (avoids dist/assets/node_modules/... which Firebase + .gitignore often skip).
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const src = path.join(
  root,
  "node_modules",
  "@expo",
  "vector-icons",
  "build",
  "vendor",
  "react-native-vector-icons",
  "Fonts",
  "Ionicons.ttf",
);
const destDir = path.join(root, "assets", "fonts");
const dest = path.join(destDir, "Ionicons.ttf");

if (!fs.existsSync(src)) {
  console.error("ensure-ionicons-font: missing source:", src);
  process.exit(1);
}
fs.mkdirSync(destDir, { recursive: true });
fs.copyFileSync(src, dest);
console.log("ensure-ionicons-font: copied to", dest);
