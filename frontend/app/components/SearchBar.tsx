import React, { useState } from "react";
import { TextInput, StyleSheet, Platform, View, Pressable } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useClientSafeDimensions } from "@/lib/useClientSafeDimensions";

interface SearchBarProps {
  value: string;
  onChangeText: (text: string) => void;
  onSubmitEditing: () => void;
  placeholder?: string;
}

const SearchBar: React.FC<SearchBarProps> = ({
  value,
  onChangeText,
  onSubmitEditing,
  placeholder = "Search manga title",
}) => {
  const { width } = useClientSafeDimensions();

  // Responsive width: full width on mobile (<600), 50% on larger screens
  const searchBarWidth = width < 600 ? width - 20 : Math.min(width * 0.5, 600);

  return (
    <View style={styles.container}>
      <TextInput
        style={[styles.input, { width: searchBarWidth }]}
        onChangeText={onChangeText}
        onSubmitEditing={onSubmitEditing}
        value={value}
        placeholder={placeholder}
        placeholderTextColor="#999"
        returnKeyType="search"
      />
      <Pressable
        style={styles.iconButton}
        onPress={onSubmitEditing}
        android_ripple={{ color: "rgba(255,255,255,0.15)" }}
        accessibilityRole="button"
      >
        <Ionicons name="search" size={20} color="#9daabb" />
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: "#1f1e32",
    borderRadius: 25,
    paddingHorizontal: 25,
    height: 48,
    marginVertical: 12,
  },
  input: {
    fontSize: 16,
    color: "#999",
    flex: 1,
    height: "100%",
    borderWidth: 0,
    outlineWidth: 0,
    ...Platform.select({
      android: {
        elevation: 2,
      },
      web: {
        outlineStyle: "none" as any,
      }
    })
  },
  iconButton: {
    padding: 8,
    borderRadius: 999,
  },
});

export default SearchBar;