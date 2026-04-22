import React from "react";
import { View, Text, Pressable, StyleSheet, ScrollView } from "react-native";
import { useRouter } from "expo-router";
import { GENRES, OTHER_TAGS, UPDATE_STATUS } from "./filter_tags";

interface GenreMenuProps {
  // Browse page single-select per row
  selectedGenreId?: string;
  onGenreChange?: (id: string) => void;
  selectedSortId?: string;
  onSortChange?: (id: string) => void;
  selectedStatusId?: string;
  onStatusChange?: (id: string) => void;
}

const GenreMenu: React.FC<GenreMenuProps> = ({
  selectedGenreId = "",
  onGenreChange,
  selectedSortId,
  onSortChange,
  selectedStatusId,
  onStatusChange
}) => {
  const router = useRouter();

  const handleGenrePress = (id: string) => {
    if (onGenreChange) {
      onGenreChange(id);
    } else {
      router.push({ pathname: "/browse", params: { genreId: id } });
    }
  };

  const handleSortPress = (id: string) => {
    if (onSortChange) onSortChange(id);
  };

  const handleStatusPress = (id: string) => {
    if (onStatusChange) onStatusChange(id);
  };
  
  return (
    <View style={styles.container}>
      {/* Genre Sort */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollViewContent}
      >
        {GENRES.map((genre) => (
          <Pressable
            key={genre.id || "all"}
            style={[styles.genreBadge, selectedGenreId === genre.id && styles.selectedBadge]}
            onPress={() => handleGenrePress(genre.id)}
          >
            <Text style={[styles.genreText, selectedGenreId === genre.id && styles.selectedText]}>
              {genre.name}
            </Text>
          </Pressable>
        ))}
      </ScrollView>

      {/* Sort */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollViewContent}
      >
        {OTHER_TAGS.map((tag) => {
          const isSelected = selectedSortId === tag.id;
          return (
            <Pressable
              key={tag.id}
              style={[styles.genreBadge, styles.sortBadge, isSelected && styles.selectedBadge]}
              onPress={() => handleSortPress(tag.id)}
            >
              <Text style={[styles.genreText, isSelected && styles.selectedText]}>
                {tag.name}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>

      {/* Status Sort */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollViewContent}
      >
        {UPDATE_STATUS.map((status) => {
          const isSelected = selectedStatusId === status.id;
          return (
            <Pressable
              key={status.id}
              style={[styles.genreBadge, styles.sortBadge, isSelected && styles.selectedBadge]}
              onPress={() => handleStatusPress(status.id)}
            >
              <Text style={[styles.genreText, isSelected && styles.selectedText]}>
                {status.name}
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
    alignItems: "flex-start", 
    flexDirection: "column",
    gap: 9
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
  sortBadge: {
    backgroundColor: "#555",
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