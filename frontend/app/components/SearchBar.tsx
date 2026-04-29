import React from "react";
import { TextInput, StyleSheet, Platform } from "react-native";
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
    <TextInput
      style={[styles.input, { width: searchBarWidth }]}
      onChangeText={onChangeText}
      onSubmitEditing={onSubmitEditing}
      value={value}
      placeholder={placeholder}
      placeholderTextColor="#999"
      returnKeyType="search"
    />
  );
};

const styles = StyleSheet.create({
  input: {
    height: 48, // Android minimum touch target
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 12,
    paddingHorizontal: 16,
    fontSize: 16,
    color: "#333",
    backgroundColor: "#FAFAFA",
    marginVertical: 12,
    ...Platform.select({
      android: {
        elevation: 2,
      },
    }),
  },
});

export default SearchBar;