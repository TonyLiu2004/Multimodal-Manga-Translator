import React, { useEffect, useState } from "react";
import { useLocalSearchParams, router } from "expo-router";
import { View, Text, ScrollView, StyleSheet } from "react-native";
import SearchBar from "../components/SearchBar";
import SearchResults from "../components/SearchResults";

interface Manga {
  id: string;
  attributes: {
    title: { [key: string]: string };
    description: { en?: string };
  };
  relationships: {
    type: string;
    attributes?: { fileName?: string };
  }[];
}

export default function SearchPage() {
  const { query } = useLocalSearchParams();
  const [searchQuery, setSearchQuery] = useState((query as string) || "");
  const [mangaList, setMangaList] = useState<Manga[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (query) {
      setSearchQuery(query as string);
      performSearch(query as string);
    }
  }, [query]);

  const performSearch = async (searchTerm: string) => {
    if (!searchTerm.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(
        `https://api.mangadex.org/manga?title=${searchTerm}&limit=20&includes[]=cover_art`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        },
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const json = await response.json();
      setMangaList(json.data || []);
    } catch (error) {
      console.error("Search error:", error);
      setMangaList([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    if (searchQuery.trim()) {
      router.setParams({ query: searchQuery });
      performSearch(searchQuery);
    }
  };

  const handleMangaPress = (manga: Manga) => {
    (router as any).push(`/manga/${manga.id}`);
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Search Manga</Text>
        <SearchBar
          value={searchQuery}
          onChangeText={setSearchQuery}
          onSubmitEditing={handleSearch}
          placeholder="Search for manga..."
        />
      </View>

      {loading ? (
        <View style={styles.center}>
          <Text>Searching...</Text>
        </View>
      ) : (
        <SearchResults mangaList={mangaList} onMangaPress={handleMangaPress} />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  header: {
    padding: 20,
    alignItems: "center",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
    color: "#333",
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 40,
  },
});
