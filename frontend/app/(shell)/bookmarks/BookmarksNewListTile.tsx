import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { BOOKMARK_GRID, bookmarkCardShadow } from "@/lib/bookmarkGridConstants";

export type BookmarksNewListTileProps = {
  creating: boolean;
  onPressCreate: () => void;
};

export default function BookmarksNewListTile({
  creating,
  onPressCreate,
}: BookmarksNewListTileProps) {
  const { tileWidth, coverWidth, coverHeight } = BOOKMARK_GRID;

  return (
    <View
      style={[
        styles.tile,
        { width: tileWidth },
        creating && styles.tileDisabled,
      ]}
    >
      <Pressable
        style={({ pressed }) => [
          styles.coverPress,
          pressed && styles.coverPressed,
        ]}
        onPress={onPressCreate}
        disabled={creating}
        accessibilityRole="button"
        accessibilityLabel="Create new reading list"
      >
        <View
          style={[styles.dashedArea, { width: coverWidth, height: coverHeight }]}
        >
          <Ionicons name="add" size={52} color="#9ca3af" />
        </View>
      </Pressable>
      <Text style={styles.title} numberOfLines={2}>
        Create new list
      </Text>
      <View style={styles.metaSpacer} />
      <View style={styles.actionsSpacer} />
    </View>
  );
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
  dashedArea: {
    alignSelf: "center",
    borderRadius: 10,
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: "#d1d5db",
    backgroundColor: "#f9fafb",
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    fontSize: 14,
    fontWeight: "700",
    color: "#111827",
    marginTop: 6,
    lineHeight: 18,
  },
  metaSpacer: {
    marginTop: 2,
    minHeight: 14,
  },
  actionsSpacer: {
    marginTop: 6,
    paddingTop: 6,
    minHeight: 28,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#f3f4f6",
  },
});
