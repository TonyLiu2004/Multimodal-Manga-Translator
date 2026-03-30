import { Text, ScrollView, StyleSheet } from "react-native";
import React, { useCallback, useEffect, useState } from "react";
import MangaCarousel from "./MangaCarousel";
import MangaCategoryList from "./MangaCategoryList";
import { Manga } from "../types/types";
import SearchBar from "./SearchBar";
import { type Href, useRouter } from "expo-router";
import type { MangaSearchListJson } from "@/lib/apiTypes";
import GenreMenu from "./GenreMenu";

import { BACKEND_URL } from "../config";

export default function Index() {
  const [searchQuery, setSearchQuery] = useState("");
  const [popularManga, setPopularManga] = useState<Manga[]>([]);
  const [recentManga, setRecentManga] = useState<Manga[]>([]);
  const [actionManga, setActionManga] = useState<Manga[]>([]);
  const [romanceManga, setRomanceManga] = useState<Manga[]>([]);
  const router = useRouter();

  const fetchBackend = useCallback(async (params: string) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/manga/search?${params}`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });
      const json = (await res.json()) as MangaSearchListJson;
      return json.data ?? [];
    } catch {
      return [];
    }
  }, []);

  const loadHomePage = useCallback(async () => {
    const [popular, recent, action, romance] = await Promise.all([
      fetchBackend("limit=10&order_by=followedCount&order_direction=desc"),
      fetchBackend("limit=6&order[latestUploadedChapter]=desc"),
      fetchBackend(
        "limit=6&includedTags[]=391b0423-d847-456f-aff0-8b0cfc03066b",
      ),
      fetchBackend(
        "limit=6&includedTags[]=423e2eae-a7a2-4a8b-ac03-a8351462d71d",
      ),
    ]);

    setPopularManga(popular);
    setRecentManga(recent);
    setActionManga(action);
    setRomanceManga(romance);
  }, [fetchBackend]);

  useEffect(() => {
    void loadHomePage();
  }, [loadHomePage]);

  const handleSearch = () => {
    const q = searchQuery.trim();
    if (!q) return;
    router.push({
      pathname: "/search",
      params: { query: q },
    } as Href);
  };

  return (
    <ScrollView
      style={{ flex: 1 }}
      contentContainerStyle={{
        flexGrow: 1,
        alignItems: "center",
        padding: 20,
      }}
    >
      <Text style={styles.h1}>Manglify</Text>

      <SearchBar
        placeholder="Search manga..."
        value={searchQuery}
        onChangeText={setSearchQuery}
        onSubmitEditing={handleSearch}
      />

      <MangaCarousel data={popularManga} />

      <GenreMenu />

      <MangaCategoryList title="Recently Updated" data={recentManga} />

      <MangaCategoryList title="Action" data={actionManga} />

      <MangaCategoryList title="Romance" data={romanceManga} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  h1: {
    fontSize: 32,
    fontWeight: "bold",
    flexShrink: 1,
  },
});
