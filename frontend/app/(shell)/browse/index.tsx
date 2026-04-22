import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  Pressable,
  useWindowDimensions,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import { BACKEND_URL } from "../../config";
import MangaBrowseCard from "@/app/components/MangaBrowseCard";
import { Manga } from "@/app/types/types";
import GenreMenu from "@/app/components/GenreMenu";
import { SORT_MAP } from "@/app/components/filter_tags";

const ITEMS_PER_PAGE = 24;

const BrowsePage = () => {
  const { genreId } = useLocalSearchParams();
  const { width } = useWindowDimensions();
  const isMobile = width < 768;

  // Reference for scrolling to top
  const flatListRef = useRef<FlatList>(null);

  const [selectedGenreId, setSelectedGenreId] = useState<string>(
    genreId ? (genreId as string) : ""
  );
  const [selectedSortId, setSelectedSortId] = useState("followedCount_desc");
  const [selectedStatusId, setSelectedStatusId] = useState<string>("");

  const [page, setPage] = useState(1);
  const [manga, setManga] = useState<Manga[]>([]);
  const [loading, setLoading] = useState(false);

  const numColumns = isMobile ? 1 : 2;
  const cardWidth = 200;
  const cardHeight = 300;

  useEffect(() => {
    fetchFilteredManga();
    flatListRef.current?.scrollToOffset({ offset: 0, animated: true });
  }, [selectedGenreId, selectedSortId, selectedStatusId, page]);

  useEffect(() => {
    if (genreId) {
      setSelectedGenreId(genreId as string);
      setPage(1);
    }
  }, [genreId]);

  const handleGenreChange = (id: string) => {
    setSelectedGenreId(id);
    setPage(1);
  };

  const handleSortChange = (id: string) => {
    setSelectedSortId(id);
    setPage(1);
  };

  const handleStatusChange = (id: string) => {
    setSelectedStatusId((prev) => (prev === id ? "" : id));
    setPage(1);
  };

  const fetchFilteredManga = async () => {
    setLoading(true);
    const offset = (page - 1) * ITEMS_PER_PAGE;

    const sort = SORT_MAP[selectedSortId] ?? { order_by: "followedCount", order_direction: "desc" };
    const params = new URLSearchParams({
      limit: ITEMS_PER_PAGE.toString(),
      offset: offset.toString(),
      order_by: sort.order_by,
      order_direction: sort.order_direction,
    });
    if (selectedGenreId) params.append("includedTags", selectedGenreId);
    if (selectedStatusId) params.append("status", selectedStatusId);
    const query = params.toString();

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
      <GenreMenu
        selectedGenreId={selectedGenreId}
        onGenreChange={handleGenreChange}
        selectedSortId={selectedSortId}
        onSortChange={handleSortChange}
        selectedStatusId={selectedStatusId}
        onStatusChange={handleStatusChange}
      />

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
          contentContainerStyle={styles.listContainer}
          columnWrapperStyle={isMobile ? undefined : styles.columnWrapper}
          renderItem={({ item }) => (
            <View style={!isMobile && styles.webCardWrapper}>
              <MangaBrowseCard manga={item} />
            </View>
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
  listContainer: { width: "100%", paddingBottom: 20 },
  columnWrapper: { gap: 0, marginBottom: 0 },
  webCardWrapper: { flex: 1 },
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
