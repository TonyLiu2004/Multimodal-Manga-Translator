import { Slot } from "expo-router";
import React from "react";
import { StyleSheet, View } from "react-native";
import SideRail from "../components/SideRail";

export default function ShellLayout() {
  return (
    <View style={styles.root}>
      <SideRail />
      <View style={styles.main}>
        <Slot />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "#f3f4f6",
  },
  main: {
    flex: 1,
    minWidth: 0,
  },
});
