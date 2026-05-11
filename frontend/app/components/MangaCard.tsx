import React, { useState } from "react";
import {
  View,
  Text,
  Image,
  Pressable,
  StyleSheet,
  Platform,
} from "react-native";
import { Manga, Chapter } from "@/lib/mangaTypes";
import PopUp from "./PopUp";
import { BACKEND_URL } from "@/config";

interface MangaCardProps {
  manga: Manga;
  width: number;
  height: number;
}

const MangaCard: React.FC<MangaCardProps> = ({ manga, width, height }) => {
  const [popupVisible, setPopupVisible] = useState(false);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loadingChapters, setLoadingChapters] = useState(false);

  const mainTitle = Object.values(manga.attributes.title)[0] || "Untitled";

  const genres = manga.attributes.tags
    ?.filter(
      (tag: { attributes: { group: string } }) =>
        tag.attributes.group === "genre",
    )
    .map(
      (tag: { attributes: { name: { en: string } } }) => tag.attributes.name.en,
    )
    .slice(0, 2);

  const coverArt = manga.relationships?.find((rel) => rel.type === "cover_art");
  const fileName = coverArt?.attributes?.fileName;
  const coverUrl = fileName
    ? `${BACKEND_URL}/api/manga/cover_art?manga_id=${manga.id}&file_name=${fileName}&size=256`
    : "https://via.placeholder.com/256x360?text=No+Cover";

  const fetchChapters = async (mangaId: string) => {
    setLoadingChapters(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/manga/${mangaId}/chapters`);
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

        {/* Title and Genre Section */}
        <View style={[styles.infoContainer, { width: width }]}>
          <Text style={styles.mangaTitle} numberOfLines={1}>
            {mainTitle}
          </Text>
          <View style={styles.genreRow}>
            {genres?.map((genre, index) => (
              <Text key={index} style={styles.genreText}>
                {genre}
                {index < genres.length - 1 ? " • " : ""}
              </Text>
            ))}
          </View>
        </View>
      </Pressable>

      <PopUp
        visible={popupVisible}
        title={mainTitle}
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
};;

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: Platform.OS === "android" ? 1 : 5,
    marginBottom: 10,
  },
  pressable: {
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
  },
  infoContainer: {
    marginTop: 8,
    alignItems: "flex-start",
  },
  mangaTitle: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#ffffff",
  },
  genreRow: {
    flexDirection: "row",
    marginTop: 2,
  },
  genreText: {
    fontSize: 11,
    color: "#a7a7a7",
  },
});

export default MangaCard;
