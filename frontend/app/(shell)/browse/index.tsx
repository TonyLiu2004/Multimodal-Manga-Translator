import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  Pressable,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import { BACKEND_URL } from "../../config";
import MangaCard from "@/app/components/MangaCard";
import { Manga } from "@/app/types/types";
import GenreMenu from "@/app/components/GenreMenu";

const ITEMS_PER_PAGE = 24;

const BrowsePage = () => {
  const { genreId } = useLocalSearchParams();

  // Reference for scrolling to top
  const flatListRef = useRef<FlatList>(null);

  const [filters, setFilters] = useState({
    genreId: genreId || "",
    status: "",
  });

  const [page, setPage] = useState(1);
  const [manga, setManga] = useState<Manga[]>([]);
  const [loading, setLoading] = useState(false);

  const numColumns = 6;
  const cardWidth = 200;
  const cardHeight = 300;

  useEffect(() => {
    fetchFilteredManga();

    // Scroll to top whenever page or filters change
    flatListRef.current?.scrollToOffset({ offset: 0, animated: true });
  }, [filters, page]);

  useEffect(() => {
    if (genreId) {
      setFilters((prev) => ({ ...prev, genreId: genreId as string }));
      setPage(1); // Reset to first page on new genre selection
    }
  }, [genreId]);

  const fetchFilteredManga = async () => {
    setLoading(true);
    const offset = (page - 1) * ITEMS_PER_PAGE;

    const query = new URLSearchParams({
      limit: ITEMS_PER_PAGE.toString(),
      offset: offset.toString(),
      "order[followedCount]": "desc",
      ...(filters.genreId && { "includedTags[]": filters.genreId as string }),
    }).toString();

    try {
      const res = await fetch(`${BACKEND_URL}/api/manga/search?${query}`, {
        headers: { "ngrok-skip-browser-warning": "true" },
      });
      const json = await res.json();
      setManga(json.data || []);
    } catch (err) {
      console.error("Fetch Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const renderPagination = () => {
    if (manga.length === 0 && !loading) return null;

    return (
      <View style={styles.paginationContainer}>
        <Pressable
          disabled={page === 1 || loading}
          onPress={() => setPage((p) => p - 1)}
          style={[
            styles.pageButton,
            (page === 1 || loading) && styles.disabledButton,
          ]}
        >
          <Text style={styles.pageButtonText}>Previous</Text>
        </Pressable>

        <View style={styles.pageIndicator}>
          <Text style={styles.pageText}>Page {page}</Text>
        </View>

        <Pressable
          disabled={loading || manga.length < ITEMS_PER_PAGE}
          onPress={() => setPage((p) => p + 1)}
          style={[
            styles.pageButton,
            (loading || manga.length < ITEMS_PER_PAGE) && styles.disabledButton,
          ]}
        >
          <Text style={styles.pageButtonText}>Next</Text>
        </Pressable>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <GenreMenu />

      {loading && page === 1 ? (
        <View style={styles.centered}>
          <ActivityIndicator size="large" color="#000" />
        </View>
      ) : (
        <FlatList
          ref={flatListRef} 
          data={manga}
          key={numColumns}
          numColumns={numColumns}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.gridContainer}
          columnWrapperStyle={styles.columnWrapper}
          renderItem={({ item }) => (
            <MangaCard manga={item} width={cardWidth} height={cardHeight} />
          )}
          ListEmptyComponent={<Text style={styles.empty}>No manga found.</Text>}
          ListFooterComponent={renderPagination}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "white" },
  centered: { flex: 1, justifyContent: "center", alignItems: "center" },
  gridContainer: { width: "100%", paddingHorizontal: 10, paddingBottom: 20 },
  columnWrapper: { justifyContent: "center", gap: 20, marginBottom: 20 },
  empty: { color: "#888", textAlign: "center", marginTop: 50 },

  paginationContainer: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
  },
  pageButton: {
    backgroundColor: "#f0f0f0",
    paddingHorizontal: 25,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#ddd",
  },
  disabledButton: {
    opacity: 0.4,
  },
  pageButtonText: {
    color: "black",
    fontWeight: "600",
    fontSize: 16,
  },
  pageIndicator: {
    minWidth: 100,
    alignItems: "center",
  },
  pageText: {
    color: "black",
    fontSize: 18,
    fontWeight: "bold",
  },
});

export default BrowsePage;
