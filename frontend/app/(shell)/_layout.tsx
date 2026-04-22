import { Slot } from "expo-router";
import React from "react";
import { StyleSheet, View, useWindowDimensions } from "react-native";
import SideRail from "../components/SideRail";
import HamburgerMenu from "../components/HamburgerMenu";

const BREAKPOINT = 768;

export default function ShellLayout() {
  const { width } = useWindowDimensions();
  const isMobile = width < BREAKPOINT;

  return (
    <View style={isMobile ? styles.mobile : styles.wide}>
      {isMobile ? <HamburgerMenu /> : <SideRail />}
      <View style={styles.main}>
        <Slot />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wide: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "#f3f4f6",
  },
  mobile: {
    flex: 1,
    flexDirection: "column",
    backgroundColor: "#f3f4f6",
  },
  main: {
    flex: 1,
    minWidth: 0,
  },
});
