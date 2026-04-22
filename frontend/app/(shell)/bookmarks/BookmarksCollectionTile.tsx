import React from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import type { ReadingListCollection } from "@/lib/readingListApi";
import {
  BOOKMARK_GRID,
  BOOKMARKS_COVER_PLACEHOLDER,
  bookmarkCardShadow,
} from "@/lib/bookmarkGridConstants";

export type BookmarksCollectionTileProps = {
  collection: ReadingListCollection;
  coverUri: string;
  coverLoading: boolean;
  isEditing: boolean;
  editName: string;
  onChangeEditName: (name: string) => void;
  onOpenList: () => void;
  onStartRename: () => void;
  onCancelRename: () => void;
  onSaveRename: () => void;
  savingRename: boolean;
  onRequestDelete: () => void;
  deleting: boolean;
};

export default function BookmarksCollectionTile({
  collection: c,
  coverUri,
  coverLoading,
  isEditing,
  editName,
  onChangeEditName,
  onOpenList,
  onStartRename,
  onCancelRename,
  onSaveRename,
  savingRename,
  onRequestDelete,
  deleting,
}: BookmarksCollectionTileProps) {
  const { tileWidth, coverWidth, coverHeight } = BOOKMARK_GRID;

  return (
    <View
      style={[styles.tile, { width: tileWidth }, deleting && styles.tileDisabled]}
    >
      <Pressable
        style={({ pressed }) => [
          styles.coverPress,
          pressed && styles.coverPressed,
        ]}
        onPress={onOpenList}
        disabled={deleting}
      >
        <View style={[styles.coverWrap, { width: coverWidth, height: coverHeight }]}>
          {coverLoading ? (
            <View style={styles.coverLoading}>
              <ActivityIndicator color="#6b7280" />
            </View>
          ) : (
            <Image
              source={{ uri: coverUri }}
              style={[styles.cover, { width: coverWidth, height: coverHeight }]}
              resizeMode="cover"
            />
          )}
          <View style={styles.badge}>
            <Text style={styles.badgeText}>{c.manga_count}</Text>
          </View>
        </View>
      </Pressable>
      <Text style={styles.listTitle} numberOfLines={2}>
        {c.name}
      </Text>
      <Text style={styles.meta}>
        {c.manga_count} title{c.manga_count === 1 ? "" : "s"}
      </Text>

      {isEditing ? (
        <View style={styles.editBlock}>
          <TextInput
            style={styles.renameInput}
            value={editName}
            onChangeText={onChangeEditName}
            autoFocus
            editable={!savingRename}
          />
          <View style={styles.editActions}>
            <Pressable
              style={({ pressed }) => [
                styles.smallPrimary,
                pressed && styles.smallPrimaryPressed,
              ]}
              onPress={() => void onSaveRename()}
              disabled={savingRename || !editName.trim()}
            >
              {savingRename ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Text style={styles.smallPrimaryText}>Save</Text>
              )}
            </Pressable>
            <Pressable
              style={({ pressed }) => [
                styles.smallGhost,
                pressed && styles.smallGhostPressed,
              ]}
              onPress={onCancelRename}
              disabled={savingRename}
            >
              <Text style={styles.smallGhostText}>Cancel</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <View style={styles.actionsRow}>
          <Pressable
            onPress={onStartRename}
            disabled={deleting}
            hitSlop={8}
          >
            <Text style={styles.link}>Rename</Text>
          </Pressable>
          <Pressable onPress={onRequestDelete} disabled={deleting} hitSlop={8}>
            <Text style={styles.danger}>
              {deleting ? "…" : "Delete"}
            </Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

export function resolveCollectionCoverUri(
  extId: string | null | undefined,
  coverUrls: Record<string, string | null>,
): { uri: string; loading: boolean } {
  const coverState =
    extId != null && extId !== "" ? coverUrls[extId] : BOOKMARKS_COVER_PLACEHOLDER;
  const loading = extId != null && extId !== "" && !(extId in coverUrls);
  const uri =
    loading || coverState == null
      ? BOOKMARKS_COVER_PLACEHOLDER
      : coverState || BOOKMARKS_COVER_PLACEHOLDER;
  return { uri, loading };
}

const styles = StyleSheet.create({
  tile: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 6,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...bookmarkCardShadow,
  },
  tileDisabled: { opacity: 0.55 },
  coverPress: { borderRadius: 10, overflow: "hidden" },
  coverPressed: { opacity: 0.92 },
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
    backgroundColor: "#e5e7eb",
  },
  badge: {
    position: "absolute",
    top: 4,
    right: 4,
    minWidth: 22,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 999,
    backgroundColor: "rgba(17,24,39,0.88)",
    alignItems: "center",
    justifyContent: "center",
  },
  badgeText: {
    fontSize: 10,
    fontWeight: "700",
    color: "#fff",
  },
  listTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "#111827",
    marginTop: 6,
    lineHeight: 18,
  },
  meta: {
    fontSize: 12,
    color: "#6b7280",
    marginTop: 2,
  },
  actionsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 6,
    paddingTop: 6,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#f3f4f6",
  },
  editBlock: {
    marginTop: 6,
    gap: 6,
  },
  link: {
    fontSize: 12,
    fontWeight: "600",
    color: "#2563eb",
  },
  danger: {
    fontSize: 12,
    fontWeight: "600",
    color: "#b91c1c",
  },
  renameInput: {
    height: 44,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 10,
    paddingHorizontal: 12,
    fontSize: 16,
    color: "#111827",
    backgroundColor: "#f9fafb",
    marginTop: 8,
  },
  editActions: {
    flexDirection: "row",
    gap: 10,
    justifyContent: "flex-end",
  },
  smallPrimary: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "#111827",
    minWidth: 72,
    alignItems: "center",
    justifyContent: "center",
  },
  smallPrimaryPressed: { opacity: 0.88 },
  smallPrimaryText: { color: "#fff", fontSize: 14, fontWeight: "600" },
  smallGhost: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    backgroundColor: "#fff",
  },
  smallGhostPressed: { opacity: 0.85 },
  smallGhostText: { color: "#374151", fontSize: 14, fontWeight: "600" },
});
