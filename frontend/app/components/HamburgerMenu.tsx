import React, { useState } from "react";
import { View, Pressable, StyleSheet, Modal, Text } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { type Href, usePathname, useRouter } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useAuth } from "@/context/AuthContext";

type NavItem = {
  label: string;
  icon: string;
  onPress: () => void;
  active: boolean;
};

export default function HamburgerMenu() {
  const [open, setOpen] = useState(false);
  const [headerHeight, setHeaderHeight] = useState(52);
  const router = useRouter();
  const pathname = usePathname() ?? "";
  const insets = useSafeAreaInsets();
  const { session, signOut } = useAuth();

  const go = (path: string) => {
    setOpen(false);
    router.replace(path as Href);
  };

  const onSignOut = async () => {
    setOpen(false);
    await signOut();
    router.replace("/sign-in" as Href);
  };

  const isHome = pathname === "/" || pathname === "";
  const isSearch = pathname === "/search" || pathname.startsWith("/search");
  const isBrowse = pathname === "/browse" || pathname.startsWith("/browse");
  const isBookmarks =
    pathname === "/bookmarks" || pathname.startsWith("/reading-list");
  const isProfile = pathname === "/profile";

  const items: NavItem[] = [
    { label: "Home", icon: "home", onPress: () => go("/"), active: isHome },
    {
      label: "Search",
      icon: "search",
      onPress: () => go("/search"),
      active: isSearch,
    },
    {
      label: "Browse",
      icon: "grid",
      onPress: () => go("/browse"),
      active: isBrowse,
    },
    ...(session
      ? [
          {
            label: "Bookmarks",
            icon: "bookmark",
            onPress: () => go("/bookmarks"),
            active: isBookmarks,
          },
          {
            label: "Profile",
            icon: "settings-outline",
            onPress: () => go("/profile"),
            active: isProfile,
          },
        ]
      : [
          {
            label: "Sign in",
            icon: "log-in-outline",
            onPress: () => go("/sign-in"),
            active: false,
          },
        ]),
  ];

  return (
    <>
      <View
        style={[styles.header, { paddingTop: Math.max(insets.top, 12) }]}
        onLayout={(e) => setHeaderHeight(e.nativeEvent.layout.height)}
      >
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={open ? "Close menu" : "Open menu"}
          onPress={() => setOpen((o) => !o)}
          style={styles.hamburger}
        >
          <Ionicons name={open ? "close" : "menu"} size={28} color="#111827" />
        </Pressable>
      </View>

      <Modal
        visible={open}
        transparent
        animationType="fade"
        onRequestClose={() => setOpen(false)}
      >
        <Pressable style={styles.backdrop} onPress={() => setOpen(false)}>
          <View
            style={[styles.dropdown, { top: headerHeight }]}
            onStartShouldSetResponder={() => true}
          >
            {items.map((item) => (
              <Pressable
                key={item.label}
                onPress={item.onPress}
                style={({ pressed }) => [
                  styles.menuItem,
                  item.active && styles.menuItemActive,
                  pressed && styles.menuItemPressed,
                ]}
              >
                <Ionicons
                  name={item.icon as any}
                  size={22}
                  color={item.active ? "#2563eb" : "#374151"}
                />
                <Text
                  style={[
                    styles.menuLabel,
                    item.active && styles.menuLabelActive,
                  ]}
                >
                  {item.label}
                </Text>
              </Pressable>
            ))}

            {session && (
              <Pressable
                onPress={() => void onSignOut()}
                style={({ pressed }) => [
                  styles.menuItem,
                  styles.signOutItem,
                  pressed && styles.menuItemPressed,
                ]}
              >
                <Ionicons name="log-out-outline" size={22} color="#6b7280" />
                <Text style={[styles.menuLabel, styles.signOutLabel]}>
                  Sign out
                </Text>
              </Pressable>
            )}
          </View>
        </Pressable>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  header: {
    backgroundColor: "#fff",
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#d1d5db",
    flexDirection: "row",
    alignItems: "flex-end",
    paddingHorizontal: 12,
    paddingBottom: 8,
  },
  hamburger: {
    padding: 6,
  },
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.18)",
  },
  dropdown: {
    position: "absolute",
    left: 0,
    right: 0,
    backgroundColor: "#fff",
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#d1d5db",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 6,
    elevation: 8,
  },
  menuItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    paddingVertical: 14,
    paddingHorizontal: 20,
  },
  menuItemActive: {
    backgroundColor: "#eff6ff",
  },
  menuItemPressed: {
    backgroundColor: "#f3f4f6",
  },
  menuLabel: {
    fontSize: 16,
    color: "#374151",
  },
  menuLabelActive: {
    color: "#2563eb",
    fontWeight: "600",
  },
  signOutItem: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#e5e7eb",
    marginTop: 4,
  },
  signOutLabel: {
    color: "#6b7280",
  },
});
