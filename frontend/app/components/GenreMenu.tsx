import React from "react";
import { View, Text, Pressable, StyleSheet, ScrollView, Platform } from "react-native";
import { useRouter } from "expo-router";

// Partial list of genres 
const GENRES = [
  { id: "", name: "All" }, // Empty ID for showing all manga
  { id: "391b0423-d847-456f-aff0-8b0cfc03066b", name: "Action" },
  { id: "423e2eae-a7a2-4a8b-ac03-a8351462d71d", name: "Romance" },
  { id: "4d32cc48-9f00-4cca-9b5a-a839f0764984", name: "Comedy" },
  { id: "b9af3a63-f058-46de-a9a0-e0c13906197a", name: "Drama" },
  { id: "cdad7e68-1419-41dd-bdce-27753074a640", name: "Horror" },
  { id: "256c8bd9-4904-4360-bf4f-508a76d67183", name: "Sci-Fi" },
];

interface GenreMenuProps {
  selectedGenreId?: string;
}

const GenreMenu: React.FC<GenreMenuProps> = ({ selectedGenreId = "" }) => {
  const router = useRouter();

  const handleGenrePress = (genreId: string) => {
    router.push({
      pathname: "/browse",
      params: { genreId },
    });
  };

  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollViewContent}
      >
        {GENRES.map((genre) => {
          const isSelected = selectedGenreId === genre.id;
          return (
            <Pressable
              key={genre.id || "all"}
              style={[styles.genreBadge, isSelected && styles.selectedBadge]}
              onPress={() => handleGenrePress(genre.id)}
            >
              <Text style={[styles.genreText, isSelected && styles.selectedText]}>
                {genre.name}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginVertical: 20,
    width: "100%",
    alignItems: "center", 
  },
  scrollViewContent: {
    flexDirection: "row",
    justifyContent: "center", 
    paddingHorizontal: 10,
  },
  genreBadge: {
    backgroundColor: "#333",
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    marginHorizontal: 5, 
  },
  genreText: {
    color: "#fff",
    fontWeight: "600",
  },
  selectedBadge: {
    backgroundColor: "#007AFF",
    borderWidth: 2,
    borderColor: "#0051D5",
  },
  selectedText: {
    fontWeight: "700",
  },
});

export default GenreMenu;