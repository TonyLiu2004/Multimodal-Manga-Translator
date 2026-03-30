import React from "react";
import { View, Text, Pressable, StyleSheet, ScrollView } from "react-native";
import { useRouter } from "expo-router";

// Partial list of genres 
const GENRES = [
  { id: "391b0423-d847-456f-aff0-8b0cfc03066b", name: "Action" },
  { id: "423e2eae-a7a2-4a8b-ac03-a8351462d71d", name: "Romance" },
  { id: "3377281d-9143-4c18-91fb-5e9373bf6259", name: "Comedy" },
  { id: "b9af3a06-8848-4c05-aee0-a1976a445a4f", name: "Drama" },
  { id: "ee06359a-b684-474a-a4a3-f0ed2d385638", name: "Horror" },
  { id: "cdad7e68-1419-41dd-a110-990ad88ee7a9", name: "Sci-Fi" },
];

const GenreMenu = () => {
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
        {GENRES.map((genre) => (
          <Pressable
            key={genre.id}
            style={styles.genreBadge}
            onPress={() => handleGenrePress(genre.id)}
          >
            <Text style={styles.genreText}>{genre.name}</Text>
          </Pressable>
        ))}
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
});

export default GenreMenu;
