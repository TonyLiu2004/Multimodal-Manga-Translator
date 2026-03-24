import React, { useState } from "react";
import {
  View,
  Image,
  Pressable,
  StyleSheet,
} from "react-native";
import { Manga, Chapter } from "../types/types";
import PopUp from "./PopUp";

const BACKEND_URL = "http://localhost:8000";

interface MangaCardProps {
  manga: Manga;
  width: number;
  height: number;
}

const MangaCard: React.FC<MangaCardProps> = ({ manga, width, height }) => {
  const [popupVisible, setPopupVisible] = useState(false);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loadingChapters, setLoadingChapters] = useState(false);

  const coverArt = manga.relationships?.find((rel) => rel.type === "cover_art");
  const fileName = coverArt?.attributes?.fileName;
  const coverUrl = fileName
    ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
    : "https://via.placeholder.com/256x360?text=No+Cover";

  const fetchChapters = async (mangaId: string) => {
    setLoadingChapters(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/manga/${mangaId}/chapters`);
      const json = await res.json();

      console.log("Raw chapter data from backend:", json);

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
    fetchChapters(manga.id);
  };

  return (
    <View style={[styles.container, { width: width + 24 }]}>
      <Pressable
        style={({ pressed }) => [
          styles.pressable,
          { opacity: pressed ? 0.8 : 1 },
        ]}
        onPress={handlePress}
      >
        <Image
          source={{ uri: coverUrl }}
          style={{ width: width, height: height, borderRadius: 10 }}
        />
      </Pressable>

      {/* The PopUp is now internal to each card */}
      <PopUp
        visible={popupVisible}
        title={Object.values(manga.attributes.title)[0] || "Untitled"}
        summary={
          manga.attributes.description?.en || "No description available."
        }
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
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 12,
  },
  pressable: {
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
    backgroundColor: "#eee", // Placeholder color while loading
  },
});

export default MangaCard;
