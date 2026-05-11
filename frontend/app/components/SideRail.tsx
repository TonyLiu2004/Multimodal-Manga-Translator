import React from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { type Href, usePathname, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useAuth } from "@/context/AuthContext";

/** Exported so other layouts can match available width beside the shell rail. */
export const SIDE_RAIL_WIDTH = 56;
const ICON_SZ = 26;
const ICON_COLOR = "#c5d0e7";
const ICON_ACTIVE = "#2563eb";

export default function SideRail() {
  const router = useRouter();
  const pathname = usePathname() ?? "";
  const insets = useSafeAreaInsets();
  const { session, signOut } = useAuth();

  const onSignOut = async () => {
    await signOut();
    router.replace("/sign-in" as Href);
  };

  const isHome = pathname === "/" || pathname === "";
  const isSearch = pathname === "/search" || pathname.startsWith("/search");
  const isBookmarks =
    pathname === "/bookmarks" || pathname.startsWith("/reading-list");
  const isProfile = pathname === "/profile";
  const isBrowse = pathname === "/browse" || pathname.startsWith("/browse"); 

  return (
    <View
      style={[
        styles.rail,
        {
          width: SIDE_RAIL_WIDTH,
          paddingTop: Math.max(insets.top, 12),
          paddingBottom: Math.max(insets.bottom, 12),
        },
      ]}
    >
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Home"
        onPress={() => router.replace("/" as Href)}
        style={({ pressed }) => [
          styles.item,
          isHome && styles.itemActive,
          pressed && styles.itemPressed,
        ]}
      >
        <Ionicons
          name="home"
          size={ICON_SZ}
          color={isHome ? ICON_ACTIVE : ICON_COLOR}
        />
      </Pressable>

      <View style={styles.underHomeGap} />

      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Search manga"
        onPress={() => router.push("/search" as Href)}
        style={({ pressed }) => [
          styles.item,
          isSearch && styles.itemActive,
          pressed && styles.itemPressed,
        ]}
      >
        <Ionicons
          name="search"
          size={ICON_SZ}
          color={isSearch ? ICON_ACTIVE : ICON_COLOR}
        />
      </Pressable>

      <View style={styles.underHomeGap} />
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Browse manga"
        onPress={() => router.push("/browse" as Href)}
        style={({ pressed }) => [
          styles.item,
          isBrowse && styles.itemActive,
          pressed && styles.itemPressed,
        ]}
      >
        <Ionicons
          name="grid" // You could also use "compass-outline" or "library-outline"
          size={ICON_SZ}
          color={isBrowse ? ICON_ACTIVE : ICON_COLOR}
        />
      </Pressable>

      {session ? (
        <>
          <View style={styles.underHomeGap} />

          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Bookmarks"
            onPress={() => router.replace("/bookmarks" as Href)}
            style={({ pressed }) => [
              styles.item,
              isBookmarks && styles.itemActive,
              pressed && styles.itemPressed,
            ]}
          >
            <Ionicons
              name="bookmark"
              size={ICON_SZ}
              color={isBookmarks ? ICON_ACTIVE : ICON_COLOR}
            />
          </Pressable>

          <View style={styles.spacer} />

          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Profile and settings"
            onPress={() => router.replace("/profile" as Href)}
            style={({ pressed }) => [
              styles.item,
              isProfile && styles.itemActive,
              pressed && styles.itemPressed,
            ]}
          >
            <Ionicons
              name="settings-outline"
              size={ICON_SZ}
              color={isProfile ? ICON_ACTIVE : ICON_COLOR}
            />
          </Pressable>

          <View style={styles.afterGearGap} />
        </>
      ) : (
        <View style={styles.spacer} />
      )}

      {session ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Sign out"
          onPress={() => void onSignOut()}
          style={({ pressed }) => [styles.item, pressed && styles.itemPressed]}
        >
          <Ionicons name="log-out-outline" size={ICON_SZ} color={ICON_COLOR} />
        </Pressable>
      ) : (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Sign in"
          onPress={() => router.push("/sign-in" as Href)}
          style={({ pressed }) => [styles.item, pressed && styles.itemPressed]}
        >
          <Ionicons name="log-in-outline" size={ICON_SZ} color={ICON_COLOR} />
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  rail: {
    backgroundColor: "#151045",
    borderRightWidth: StyleSheet.hairlineWidth,
    borderRightColor: "#0e0935",
    alignItems: "center",
  },
  underHomeGap: {
    height: 10,
  },
  afterGearGap: {
    height: 10,
  },
  spacer: {
    flex: 1,
    minHeight: 24,
  },
  item: {
    width: 44,
    height: 44,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  itemActive: {
    backgroundColor: "#eff6ff",
  },
  itemPressed: {
    opacity: 0.75,
  },
});
