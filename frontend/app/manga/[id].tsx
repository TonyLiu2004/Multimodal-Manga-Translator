import { useLocalSearchParams } from "expo-router";
import {
  View,
  Text,
  Image,
  ScrollView,
  StyleSheet,
  Pressable,
} from "react-native";
import React, { useEffect, useState } from "react";

interface Chapter {
  id: string;
  chapter: string;
  title: string;
  pages: number;
}

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

export default function MangaDetails() {
  const { id } = useLocalSearchParams();
  const [manga, setManga] = useState<Manga | null>(null);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) {
      fetchMangaDetails(id as string);
      fetchChapters(id as string);
    }
  }, [id]);

  const fetchMangaDetails = async (mangaId: string) => {
    try {
      const response = await fetch(
        `https://api.mangadex.org/manga/${mangaId}?includes[]=cover_art`,
      );
      const data = await response.json();
      setManga(data.data);
    } catch (error) {
      console.error("Error fetching manga details:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchChapters = async (mangaId: string) => {
    try {
      const response = await fetch(
        `https://api.mangadex.org/manga/${mangaId}/feed?limit=6&order[chapter]=asc&translatedLanguage[]=en`,
      );
      const data = await response.json();
      const chapterData: Chapter[] = (data.data || []).map((ch: any) => ({
        id: ch.id,
        chapter: ch.attributes.chapter || "?",
        title: ch.attributes.title || "",
        pages: ch.attributes.pages || 0,
      }));
      setChapters(chapterData);
    } catch (error) {
      console.error("Error fetching chapters:", error);
    }
  };

  if (loading || !manga) {
    return (
      <View style={styles.center}>
        <Text>Loading...</Text>
      </View>
    );
  }

  const titles = Object.values(manga.attributes.title);
  const displayTitle = (titles[0] as string) || "Untitled";

  const coverArt = manga.relationships.find(
    (rel: any) => rel.type === "cover_art",
  );
  const fileName = coverArt?.attributes?.fileName;
  const coverUrl = fileName
    ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
    : "https://via.placeholder.com/256x360?text=No+Cover";

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Image source={{ uri: coverUrl }} style={styles.cover} />
        <View style={styles.info}>
          <Text style={styles.title}>{displayTitle}</Text>
          <Text style={styles.description}>
            {manga.attributes.description.en || "No description available."}
          </Text>
        </View>
      </View>

      <View style={styles.chaptersSection}>
        <Text style={styles.sectionTitle}>Chapters</Text>
              {chapters.length > 0 ? (
                  chapters.map((chapter) => (
                      <Pressable key={chapter.id} style={styles.chapterItem}>
                          <Text style={styles.chapterText}>
                              Chapter {chapter.chapter}
                              {chapter.title && ` - ${chapter.title}`}
                          </Text>
                          <Text style={styles.pagesText}>{chapter.pages} pages</Text>
                      </Pressable>
                  ))
              ) : (
                  <Text>No chapters available.</Text>
              )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  header: {
    flexDirection: "row",
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#e0e0e0",
  },
  cover: {
    width: 150,
    height: 225,
    borderRadius: 10,
  },
  info: {
    flex: 1,
    marginLeft: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 10,
  },
  description: {
    fontSize: 16,
    lineHeight: 24,
    color: "#666",
  },
  chaptersSection: {
    padding: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 15,
  },
  chapterItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 15,
    backgroundColor: "#f9f9f9",
    marginBottom: 8,
    borderRadius: 8,
  },
  chapterText: {
    fontSize: 16,
    flex: 1,
  },
  pagesText: {
    fontSize: 14,
    color: "#666",
  },
});
