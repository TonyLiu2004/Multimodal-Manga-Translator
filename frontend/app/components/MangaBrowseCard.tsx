import React, { useState, useEffect } from "react";
import { View, Image, Text, Pressable, StyleSheet } from "react-native";
import { Manga, Chapter } from "../../lib/mangaTypes";
import PopUp from "./PopUp";
import { BACKEND_URL } from "@/config";

interface MangaBrowseCardProps {
  manga: Manga;
}

const MangaBrowseCard: React.FC<MangaBrowseCardProps> = ({ manga }) => {
  const [popupVisible, setPopupVisible] = useState(false);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loadingChapters, setLoadingChapters] = useState(false);

  const coverRel = manga.relationships?.find((r: { type: string }) => r.type === "cover_art");
  const fileName = coverRel?.attributes?.fileName;
  const coverUrl = fileName
    ? `${BACKEND_URL}/api/manga/cover_art?manga_id=${manga.id}&file_name=${fileName}&size=256`
    : "https://via.placeholder.com/80x110?text=No+Cover";

  const title = Object.values(manga.attributes.title)[0] || "Untitled";
  const description = manga.attributes.description?.en?.slice(0, 200).trimEnd() || "";
  const descriptionTrimmed =
    description.length < (manga.attributes.description?.en?.length ?? 0)
      ? description + "…"
      : description;

  const fetchChapters = async () => {
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
    } catch {
      setChapters([]);
    } finally {
      setLoadingChapters(false);
    }
  };

  const handlePress = () => {
    setPopupVisible(true);
    fetchChapters();
  };

  return (
    <>
      <Pressable
        onPress={handlePress}
        style={({ pressed }) => [styles.card, pressed && styles.pressed]}
      >
        <Image source={{ uri: coverUrl }} style={styles.cover} />
        <View style={styles.info}>
          <Text style={styles.title} numberOfLines={2}>{title}</Text>
          {descriptionTrimmed ? (
            <Text style={styles.description} numberOfLines={4}>
              {descriptionTrimmed}
            </Text>
          ) : null}


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
    </>
  );
};

const styles = StyleSheet.create({
  card: {
    flexDirection: "row",
    alignItems: "flex-start",
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#e5e7eb",
    backgroundColor: "#fff",
    minHeight: 200,
  },
  pressed: {
    backgroundColor: "#f3f4f6",
  },
  cover: {
    width: 170,
    height: 220,
    borderRadius: 8,
    backgroundColor: "#e5e7eb",
    flexShrink: 0,
  },
  info: {
    flex: 1,
    marginLeft: 14,
    justifyContent: "flex-start",
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: "#111827",
    marginBottom: 6,
  },
  description: {
    fontSize: 15,
    color: "#6b7280",
    lineHeight: 19,
    marginBottom: 8,
  },

});

export default MangaBrowseCard;
