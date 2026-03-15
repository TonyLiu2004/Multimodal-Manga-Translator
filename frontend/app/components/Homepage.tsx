import { Text, ScrollView, StyleSheet } from "react-native";
import React, { useEffect, useState } from "react";
import { router } from "expo-router";
import SearchBar from "./SearchBar";
import Carousel from "./Carousel";
import MangaCategoryList from "./MangaCategoryList";

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

export default function Index() {
  const [searchQuery, setSearchQuery] = useState("");

  const [popularManga, setPopularManga] = useState<Manga[]>([]);
  const [actionManga, setActionManga] = useState<Manga[]>([]);
  const [romanceManga, setRomanceManga] = useState<Manga[]>([]);
  const [recentManga, setRecentManga] = useState<Manga[]>([]);

  useEffect(() => {
    async function loadHomePage() {
      const [popular, action, romance, recent] = await Promise.all([
        fetchManga("limit=8&order[followedCount]=desc"),
        fetchManga(
          "limit=6&includedTags[]=391b0423-d847-456f-aff0-8b0cfc03066b&order[followedCount]=desc",
        ),
        fetchManga(
          "limit=6&includedTags[]=423e2eae-a7a2-4a8b-ac03-a8351462d71d&order[followedCount]=desc",
        ),
        fetchManga("limit=6&order[latestUploadedChapter]=desc"),
      ]);

      setPopularManga(popular);
      setActionManga(action);
      setRomanceManga(romance);
      setRecentManga(recent);
    }

    loadHomePage();
  }, []);

  const fetchManga = async (params: string) => {
    try {
      const response = await fetch(
        `https://api.mangadex.org/manga?${params}&includes[]=cover_art`,
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.data || [];
    } catch (error) {
      console.error("Error fetching manga:", error);
      return [];
    }
  };

  const handleSearch = async () => {
    if (searchQuery.trim()) {
      // Navigate to search page with query
      (router as any).push(`/search?query=${encodeURIComponent(searchQuery)}`);
    }
  };

  const handleMangaPress = (manga: Manga) => {
    // Navigate to manga details page
    (router as any).push(`/manga/${manga.id}`);
  };

  return (
    <ScrollView
      contentContainerStyle={{ flexGrow: 1, alignItems: "center", padding: 20 }}
    >
      <Text style={styles.h1}>Manglify</Text>
      <SearchBar
        value={searchQuery}
        onChangeText={setSearchQuery}
        onSubmitEditing={handleSearch}
      />

      <Carousel data={popularManga} onMangaPress={handleMangaPress} />

      <MangaCategoryList
        title="Recently Updated"
        data={recentManga}
        onMangaPress={handleMangaPress}
      />

      <MangaCategoryList
        title="Action"
        data={actionManga}
        onMangaPress={handleMangaPress}
      />

      <MangaCategoryList
        title="Romance"
        data={romanceManga}
        onMangaPress={handleMangaPress}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  h1: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#333",
    marginVertical: 10,
  },
});
