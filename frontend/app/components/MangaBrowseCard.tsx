import React, { useState, useEffect } from "react";
import {
  View,
  Image,
  Text,
  Pressable,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { Manga, Chapter } from "../types/types";
import PopUp from "./PopUp";
import { BACKEND_URL } from "../config";

interface MangaBrowseCardProps {
  manga: Manga;
}

const MangaBrowseCard: React.FC<MangaBrowseCardProps> = ({ manga }) => {
  const [popupVisible, setPopupVisible] = useState(false);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loadingChapters, setLoadingChapters] = useState(false);
  const [recentChapters, setRecentChapters] = useState<Chapter[]>([]);

  const coverArt = manga.relationships?.find((rel) => rel.type === "cover_art");
  const fileName = coverArt?.attributes?.fileName;
  const coverUrl = fileName
    ? `${BACKEND_URL}/api/manga/cover_art?manga_id=${manga.id}&file_name=${fileName}&size=256`
    : "https://via.placeholder.com/150x200?text=No+Cover";

  const title = Object.values(manga.attributes.title)[0] || "Untitled";

  useEffect(() => {
    fetchRecentChapters();
  }, [manga.id]);

  const fetchRecentChapters = async () => {
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/manga/${manga.id}/chapters?limit=3`
      );
      const json = await res.json();

      const data = (json.data || []).slice(0, 3).map((ch: any) => ({
        id: ch.id,
        chapter: ch.attributes.chapter || "?",
        title: ch.attributes.title || "",
        pages: ch.attributes.pages || 0,
        language: ch.attributes.translatedLanguage || "unknown",
      }));

      setRecentChapters(data);
    } catch (error) {
      console.error("Error fetching recent chapters:", error);
    }
  };

  const fetchAllChapters = async () => {
    setLoadingChapters(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/manga/${manga.id}/chapters`);
      const json = await res.json();

      const data = (json.data || []).map((ch: any) => ({
        id: ch.id,
        chapter: ch.attributes.chapter || "?",
        title: ch.attributes.title || "",
        pages: ch.attributes.pages || 0,
        language: ch.attributes.translatedLanguage || "unknown",
      }));

      setChapters(data);
    } catch (error) {
      console.error("Error fetching chapters:", error);
      setChapters([]);
    } finally {
      setLoadingChapters(false);
    }
  };

  const handlePress = () => {
    setPopupVisible(true);
    fetchAllChapters();
  };

  return (
    <View style={styles.container}>
      <Pressable
        style={({ pressed }) => [
          styles.card,
          { opacity: pressed ? 0.7 : 1 },
        ]}
        onPress={handlePress}
      >
        {/* Cover Image */}
        <Image source={{ uri: coverUrl }} style={styles.cover} />

        {/* Content Section */}
        <View style={styles.content}>
          <Text style={styles.title} numberOfLines={2}>
            {title}
          </Text>

          <View style={styles.chaptersSection}>
            <Text style={styles.chaptersLabel}>Recent Chapters:</Text>
            {recentChapters.length > 0 ? (
              recentChapters.map((chapter, index) => (
                <Text key={chapter.id} style={styles.chapterText} numberOfLines={1}>
                  Ch. {chapter.chapter}
                  {chapter.title ? `: ${chapter.title}` : ""}
                </Text>
              ))
            ) : (
              <Text style={styles.noChapters}>No chapters available</Text>
            )}
          </View>
        </View>
      </Pressable>

      <PopUp
        visible={popupVisible}
        title={title}
        summary={manga.attributes.description?.en || "No description available."}
        coverArt={coverUrl}
        manga={manga}
        chapters={chapters}
        loadingChapters={loadingChapters}
        onClose={() => setPopupVisible(false)}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 16,
    marginVertical: 8,
  },
  card: {
    flexDirection: "row",
    backgroundColor: "#fff",
    borderRadius: 12,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  cover: {
    width: 100,
    height: 150,
    resizeMode: "cover",
  },
  content: {
    flex: 1,
    padding: 12,
    justifyContent: "space-between",
  },
  title: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 8,
  },
  chaptersSection: {
    flex: 1,
  },
  chaptersLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: "#666",
    marginBottom: 4,
  },
  chapterText: {
    fontSize: 12,
    color: "#555",
    marginBottom: 2,
  },
  noChapters: {
    fontSize: 12,
    color: "#999",
    fontStyle: "italic",
  },
});

export default MangaBrowseCard;