import React from "react";
import {
  ActivityIndicator,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import type { ReadingListItem } from "@/lib/readingListApi";

/** Same footprint as bookmarks / MangaCard tiles. */
const COVER_W = 200;
const COVER_H = 300;
export const READING_LIST_GRID = {
  coverWidth: COVER_W,
  coverHeight: COVER_H,
  tileWidth: COVER_W + 24,
} as const;

export type ReadingListGridTileProps = {
  item: ReadingListItem;
  coverUri: string;
  displayTitle: string;
  descLoading: boolean;
  previewText: string;
  showReadMore: boolean;
  extId: string | null;
  canOpen: boolean;
  openingThisTile: boolean;
  readerBusy: boolean;
  removing: boolean;
  onCoverPress: () => void;
  onChapterPress: () => void;
  onTitlePress: () => void;
  onReadMore: () => void;
  onRemove: () => void;
};

const cardShadow = Platform.select({
  ios: {
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  android: { elevation: 3 },
  default: {},
});

export default function ReadingListGridTile({
  item,
  coverUri,
  displayTitle,
  descLoading,
  previewText,
  showReadMore,
  extId,
  canOpen,
  openingThisTile,
  readerBusy,
  removing,
  onCoverPress,
  onChapterPress,
  onTitlePress,
  onReadMore,
  onRemove,
}: ReadingListGridTileProps) {
  const { coverWidth, coverHeight, tileWidth } = READING_LIST_GRID;

  const chapterLabel = canOpen
    ? item.last_chapter_number != null
      ? `Ch. ${item.last_chapter_number}`
      : "Ch. 1"
    : "Reader unavailable";

  return (
    <View
      style={[
        styles.tile,
        { width: tileWidth },
        removing && styles.tileDisabled,
      ]}
    >
      <Pressable
        style={({ pressed }) => [
          styles.coverPress,
          pressed && canOpen && styles.coverPressed,
          !canOpen && styles.coverDisabled,
        ]}
        onPress={onCoverPress}
        disabled={readerBusy || !canOpen}
      >
        <View style={[styles.coverWrap, { width: coverWidth, height: coverHeight }]}>
          <Image
            source={{ uri: coverUri }}
            style={[styles.cover, { width: coverWidth, height: coverHeight }]}
            resizeMode="cover"
          />
          {openingThisTile ? (
            <View style={styles.coverLoading}>
              <ActivityIndicator color="#fff" />
            </View>
          ) : null}
        </View>
      </Pressable>

      {extId ? (
        <Pressable
          style={({ pressed }) => [
            styles.titlePressable,
            pressed && styles.titlePressed,
          ]}
          onPress={onTitlePress}
          hitSlop={{ top: 4, bottom: 4 }}
          accessibilityRole="button"
          accessibilityLabel="Open manga details"
        >
          <Text
            style={styles.title}
            numberOfLines={1}
            ellipsizeMode="tail"
          >
            {displayTitle}
          </Text>
        </Pressable>
      ) : (
        <Text
          style={styles.title}
          numberOfLines={1}
          ellipsizeMode="tail"
        >
          {displayTitle}
        </Text>
      )}

      {descLoading ? (
        <View style={styles.descSpinnerWrap}>
          <ActivityIndicator size="small" color="#9ca3af" />
        </View>
      ) : (
        <Text
          style={styles.description}
          numberOfLines={showReadMore ? 4 : 6}
        >
          {previewText}
        </Text>
      )}

      {showReadMore && extId ? (
        <Pressable
          style={({ pressed }) => [
            styles.readMore,
            pressed && styles.readMorePressed,
          ]}
          onPress={onReadMore}
          hitSlop={6}
        >
          <Text style={styles.readMoreLabel}>Read more</Text>
        </Pressable>
      ) : null}

      <View style={styles.lastReadRow}>
        <View style={styles.lastReadLeft}>
          <Text style={styles.lastReadLabel}>Last read:</Text>
          <Pressable
            style={({ pressed }) => [
              styles.chapterHit,
              (!canOpen || readerBusy) && styles.chapterDisabled,
              pressed && canOpen && !readerBusy && styles.chapterPressed,
            ]}
            onPress={onChapterPress}
            disabled={!canOpen || readerBusy}
            hitSlop={{ top: 4, bottom: 4 }}
          >
            {openingThisTile ? (
              <ActivityIndicator size="small" color="#2563eb" />
            ) : (
              <Text style={styles.chapterText}>{chapterLabel}</Text>
            )}
          </Pressable>
        </View>
        <Pressable
          style={({ pressed }) => [styles.trash, pressed && styles.trashPressed]}
          onPress={onRemove}
          disabled={removing}
          accessibilityRole="button"
          accessibilityLabel="Remove from list"
          hitSlop={10}
        >
          {removing ? (
            <ActivityIndicator size="small" color="#b91c1c" />
          ) : (
            <Ionicons name="trash-outline" size={18} color="#b91c1c" />
          )}
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  tile: {
    position: "relative",
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 6,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...cardShadow,
  },
  tileDisabled: { opacity: 0.55 },
  coverPress: { borderRadius: 10, overflow: "hidden" },
  coverPressed: { opacity: 0.92 },
  coverDisabled: { opacity: 0.72 },
  coverWrap: {
    alignSelf: "center",
    borderRadius: 10,
    overflow: "hidden",
    backgroundColor: "#e5e7eb",
    position: "relative",
  },
  cover: { borderRadius: 10 },
  coverLoading: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(17,24,39,0.35)",
  },
  titlePressable: {
    alignSelf: "stretch",
    borderRadius: 6,
  },
  titlePressed: { opacity: 0.75 },
  title: {
    alignSelf: "stretch",
    maxWidth: "100%",
    flexShrink: 1,
    marginTop: 6,
    fontSize: 14,
    fontWeight: "700",
    color: "#111827",
    lineHeight: 18,
  },
  description: {
    fontSize: 12,
    color: "#4b5563",
    lineHeight: 17,
    marginTop: 4,
  },
  descSpinnerWrap: {
    marginTop: 8,
    minHeight: 40,
    justifyContent: "flex-start",
  },
  readMore: {
    alignSelf: "flex-start",
    marginTop: 4,
    paddingVertical: 2,
  },
  readMorePressed: { opacity: 0.65 },
  readMoreLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#2563eb",
  },
  lastReadRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 8,
    gap: 8,
    minHeight: 32,
  },
  lastReadLeft: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 6,
    minWidth: 0,
    marginRight: 4,
  },
  lastReadLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: "#6b7280",
  },
  chapterHit: {
    flexShrink: 1,
    paddingVertical: 2,
  },
  chapterPressed: { opacity: 0.75 },
  chapterDisabled: { opacity: 0.45 },
  chapterText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#2563eb",
  },
  trash: {
    padding: 6,
    borderRadius: 10,
    flexShrink: 0,
  },
  trashPressed: {
    opacity: 0.7,
    backgroundColor: "#fef2f2",
  },
});
