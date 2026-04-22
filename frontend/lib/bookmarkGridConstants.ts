import { Platform } from "react-native";

export const BOOKMARKS_COVER_PLACEHOLDER =
  "https://via.placeholder.com/200x300?text=List";

const COVER_W = 200;
const COVER_H = 300;

/** Same footprint as reading-list grid / MangaCard. */
export const BOOKMARK_GRID = {
  coverWidth: COVER_W,
  coverHeight: COVER_H,
  tileWidth: COVER_W + 24,
} as const;

export const bookmarkCardShadow = Platform.select({
  ios: {
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  android: { elevation: 3 },
  default: {},
});
