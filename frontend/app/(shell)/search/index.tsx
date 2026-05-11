import React, { useCallback, useEffect, useState } from "react";
import {
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import SearchBar from "@/app/components/SearchBar";
import SearchResults from "@/app/components/SearchResults";
import { BACKEND_URL } from "@/config";
import type { Manga } from "@/lib/mangaTypes";
import type { MangaSearchListJson } from "@/lib/apiTypes";

function searchParamToString(v: string | string[] | undefined): string {
  if (v == null) return "";
  return Array.isArray(v) ? (v[0] ?? "") : v;
}

export default function SearchPage() {
  const { query } = useLocalSearchParams<{ query?: string | string[] }>();
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState(() =>
    searchParamToString(query),
  );
  const [mangaList, setMangaList] = useState<Manga[]>([]);
  const [loading, setLoading] = useState(false);

  const performSearch = useCallback(async (searchTerm: string) => {
    if (!searchTerm.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(
        `${BACKEND_URL}/api/manga/search?title=${encodeURIComponent(
          searchTerm.trim(),
        )}&limit=20&includes[]=cover_art`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
          },
        },
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const json = (await response.json()) as MangaSearchListJson;
      setMangaList(json.data ?? []);
    } catch (error) {
      console.error("Search error:", error);
      setMangaList([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const q = searchParamToString(query);
    if (!q) return;
    setSearchQuery(q);
    void performSearch(q);
  }, [query, performSearch]);

  const handleSearch = () => {
    const trimmed = searchQuery.trim();
    if (!trimmed) return;
    router.setParams({ query: trimmed });
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
        <SearchResults mangaList={mangaList} />
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#33384e",
  },
  header: {
    padding: 20,
    alignItems: "center",
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
    color: "#fff",
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 40,
  },
});
