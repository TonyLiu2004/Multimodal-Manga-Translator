import React from "react";
import {
  View,
  Text,
  Pressable,
  useWindowDimensions,
  StyleSheet,
} from "react-native";
import MangaCard from "./MangaCard";
import { Manga } from "@/lib/mangaTypes";

interface SearchResultsProps {
  mangaList: Manga[];
}

const SearchResults: React.FC<SearchResultsProps> = ({
  mangaList,
}) => {
  const { width } = useWindowDimensions();
  const isDesktop = width > 600;

  if (mangaList.length === 0) return null;

  return (
    <View style={styles.container}>
      <Text style={styles.resultsCount}>{mangaList.length} manga found!</Text>

      <View style={styles.listContainer}>
        {mangaList.map((manga) => (
          <Pressable
            key={manga.id}
            style={[
              styles.mangaRow,
              {
                flexDirection: isDesktop ? "row" : "column",
                width: isDesktop ? "60%" : "90%",
              },
            ]}
          >
            <MangaCard
              manga={manga}
              width={200}
              height={300}
            />

            <View
              style={[
                styles.infoContainer,
                {
                  marginLeft: isDesktop ? 30 : 0,
                  marginTop: isDesktop ? 0 : 15,
                },
              ]}
            >
              <Text style={styles.title}>
                {Object.values(manga.attributes.title)[0] || "Untitled"}
              </Text>
              <Text style={styles.idText}>ID: {manga.id}</Text>
              <Text
                numberOfLines={isDesktop ? 8 : 4}
                style={styles.description}
              >
                {manga.attributes.description.en || "No description available."}
              </Text>
            </View>
          </Pressable>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginTop: 20,
    alignItems: "center",
    width: "100%",
  },
  resultsCount: {
    fontSize: 16,
    color: "#666",
    marginBottom: 10,
  },
  listContainer: {
    width: "100%",
    alignItems: "center",
  },
  mangaRow: {
    marginBottom: 30,
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 10,
    // Add a slight shadow for better UI
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  infoContainer: {
    flex: 1,
    justifyContent: "flex-start",
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 5,
  },
  idText: {
    color: "gray",
    fontSize: 12,
    marginBottom: 10,
  },
  description: {
    lineHeight: 20,
    color: "#333",
  },
});

export default SearchResults;
