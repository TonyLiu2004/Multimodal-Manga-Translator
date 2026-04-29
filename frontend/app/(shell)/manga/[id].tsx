import React, { useEffect, useState } from "react";
import { useLocalSearchParams, useRouter, Href } from "expo-router";
import {
  View,
  Text,
  ScrollView,
  Image,
  Pressable,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { BACKEND_URL } from "@/config";
import { Manga, Chapter } from "@/lib/mangaTypes";
import { parseChapterNumber } from "@/lib/readingListDetailManga";
import { useAuth } from "@/context/AuthContext";
import {
  addToReadingList,
  fetchReadingLists,
  ReadingListCollection,
  createReadingList,
} from "@/lib/readingListApi";

export default function MangaDetailsPage() {
  const { id } = useLocalSearchParams();
  const router = useRouter();
  const seriesId = Array.isArray(id) ? id[0] : id;
  const { session } = useAuth();

  const [manga, setManga] = useState<any | null>(null);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [coverUrl, setCoverUrl] = useState<string | null>(null);
  
  const [listBusy, setListBusy] = useState(false);
  const [listMsg, setListMsg] = useState<string | null>(null);
  const [readingLists, setReadingLists] = useState<ReadingListCollection[]>([]);
  const [listsLoading, setListsLoading] = useState(false);
  const [selectedListId, setSelectedListId] = useState<number | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<string>("All");

  useEffect(() => {
    if (id) {
      fetchMangaData();
      fetchCoverUrl();
    }
  }, [id]);

  useEffect(() => {
    if (!session?.access_token) return;
    (async () => {
      setListsLoading(true);
      try {
        let cols = await fetchReadingLists(session.access_token);
        if (cols.length === 0) {
          await createReadingList(session.access_token, "My list");
          cols = await fetchReadingLists(session.access_token);
        }
        setReadingLists(cols);
        setSelectedListId(cols[0]?.id ?? null);
      } catch (e) {
        setReadingLists([]);
      } finally {
        setListsLoading(false);
      }
    })();
  }, [session?.access_token]);

  const fetchMangaData = async () => {
    setLoading(true);
    try {
      const [mRes, cRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/manga/${id}/info`),
        fetch(`${BACKEND_URL}/api/manga/${id}/chapters`),
      ]);

      const mJson = await mRes.json();
      const cJson = await cRes.json();

      setManga(mJson.data || mJson);

      const mapped = (cJson.data || []).map((ch: any) => ({
        id: ch.id,
        chapter: ch.attributes?.chapter || "?",
        title: ch.attributes?.title || "",
        pages: ch.attributes?.pages || 0,
        language: ch.attributes?.translatedLanguage || "en",
      }));
      setChapters(mapped);
    } catch (e) {
      console.error("Fetch Error:", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchCoverUrl = async () => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/manga/${id}/cover`);
    const json = await res.json();
    
    if (json.cover_url) {
      const proxiedUrl = `${BACKEND_URL}/api/proxy/image?target_url=${encodeURIComponent(json.cover_url)}`;
      setCoverUrl(proxiedUrl);
    }
  } catch (e) {
    console.error("Failed to fetch cover JSON", e);
  }
};

  if (loading || !manga) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  const displayTitle = manga.title || "Untitled";
  const displaySummary = manga.description || "No description.";
  const availableLangs = manga.availableTranslatedLanguages || [];

  const filteredChapters = chapters.filter(
    (ch) => selectedLanguage === "All" || ch.language === selectedLanguage
  );

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        {coverUrl ? (
          <Image 
            source={{ uri: coverUrl }} 
            style={styles.coverImage}
          />
        ) : (
          <View style={[styles.coverImage, { backgroundColor: '#ccc', justifyContent: 'center', alignItems: 'center' }]}>
            <ActivityIndicator size="small" color="#999" />
          </View>
        )}
        <View style={styles.info}>
          <Text style={styles.title}>{displayTitle}</Text>
          <Text style={styles.summary}>{displaySummary}</Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.listPickerLabel}>Add to Reading List</Text>
        {!session ? (
          <Pressable style={({ pressed }) => [
                    styles.signInCtaBtn,
                    pressed && styles.signInCtaBtnPressed,
                  ]} 
                  onPress={() => router.push("/sign-in" as Href)}>
            <Text style={styles.signInCtaBtnText}>Sign in to Bookmark</Text>
          </Pressable>
        ) : (
          <>
            <View style={styles.listChipsWrap}>
              {readingLists.map((c) => (
                <Pressable
                  key={c.id}
                  style={[styles.listChip, selectedListId === c.id && styles.listChipActive]}
                  onPress={() => setSelectedListId(c.id)}
                >
                  <Text style={[styles.listChipText, selectedListId === c.id && styles.listChipTextActive]}>
                    {c.name}
                  </Text>
                </Pressable>
              ))}
            </View>
            <Pressable 
              style={[styles.addListBtn, (listBusy || !selectedListId) && styles.disabledBtn]} 
              disabled={listBusy || !selectedListId}
              onPress={async () => {
                setListBusy(true);
                try {
                  await addToReadingList(session.access_token!, {
                    readingListId: selectedListId!,
                    external_manga_id: manga.id,
                    manga_title: displayTitle,
                  });
                  setListMsg("Successfully added!");
                } catch (e) {
                  setListMsg("Failed to add.");
                } finally {
                  setListBusy(false);
                }
              }}
            >
              {listBusy ? <ActivityIndicator color="#fff" /> : <Text style={styles.addListBtnText}>Confirm Add</Text>}
            </Pressable>
            {listMsg && <Text style={styles.listMsg}>{listMsg}</Text>}
          </>
        )}
      </View>

      <View style={styles.chaptersSection}>
        <Text style={styles.chaptersTitle}>Chapters</Text>
        <Text style={styles.languageLabel}>Filter by Language:</Text>
        <View style={styles.languageContainer}>
          {["All", ...availableLangs].map((lang) => (
            <Pressable
              key={lang}
              style={[styles.langChip, selectedLanguage === lang && styles.activeLangChip]}
              onPress={() => setSelectedLanguage(lang)}
            >
              <Text style={[styles.langChipText, selectedLanguage === lang && styles.activeLangChipText]}>
                {lang.toUpperCase()}
              </Text>
            </Pressable>
          ))}
        </View>

        <View style={styles.chaptersList}>
          {filteredChapters.length > 0 ? (
            filteredChapters.map((chapter) => (
              <Pressable
                key={chapter.id}
                style={styles.chapterItem}
                onPress={() => {
                  const q = new URLSearchParams();
                  if (seriesId != null && seriesId !== "") {
                    q.set("seriesId", String(seriesId));
                  }
                  const chNum = parseChapterNumber(chapter.chapter);
                  if (chNum != null) {
                    q.set("chapterNumber", String(chNum));
                  }
                  const qs = q.toString();
                  router.push(
                    (qs
                      ? `/reader/${chapter.id}?${qs}`
                      : `/reader/${chapter.id}`) as Href,
                  );
                }}
              >
                <View style={styles.chapterRow}>
                  <Text style={styles.chapterNumber}>Ch. {chapter.chapter}</Text>
                  <Text style={styles.langTag}>{chapter.language.toUpperCase()}</Text>
                </View>
                <Text style={styles.chapterTitle}>{chapter.title || "Untitled"}</Text>
                <Text style={styles.chapterPages}>{chapter.pages} pages</Text>
              </Pressable>
            ))
          ) : (
            <Text style={styles.noChapters}>No chapters found for this language.</Text>
          )}
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff", padding: 20 },
  centered: { flex: 1, justifyContent: "center", alignItems: "center" },
  header: { flexDirection: "row", marginBottom: 25 },
  coverImage: { width: 120, height: 180, borderRadius: 8, backgroundColor: '#eee' },
  info: { flex: 1, marginLeft: 15 },
  title: { fontSize: 20, fontWeight: "bold", marginBottom: 8, color: "#111" },
  summary: { fontSize: 14, color: "#666", lineHeight: 20 },
  card: { padding: 15, backgroundColor: "#f4f4f4", borderRadius: 12, marginBottom: 20 },
  listPickerLabel: { fontSize: 15, fontWeight: "600", marginBottom: 10 },
  listChipsWrap: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginBottom: 15 },
  listChip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, backgroundColor: "#fff", borderWidth: 1, borderColor: "#ddd" },
  listChipActive: { backgroundColor: "#111", borderColor: "#111" },
  listChipText: { fontSize: 12 },
  listChipTextActive: { color: "#fff" },
  addListBtn: { backgroundColor: "#111", padding: 12, borderRadius: 8, alignItems: "center" },
  disabledBtn: { opacity: 0.5 },
  addListBtnText: { color: "#fff", fontWeight: "600" },
  signInCtaBtn: { backgroundColor: "#111", paddingVertical: 12, paddingHorizontal: 16, borderRadius: 8, alignItems: "center", alignSelf: "stretch", },
  signInCtaBtnPressed: { opacity: 0.88 },
  signInCtaBtnText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  listMsg: { marginTop: 10, textAlign: "center", color: "#2e7d32" },
  chaptersSection: { marginTop: 10 },
  chaptersTitle: { fontSize: 22, fontWeight: "bold", marginBottom: 15 },
  languageLabel: { fontSize: 14, color: "#666", marginBottom: 10 },
  languageContainer: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginBottom: 20 },
  langChip: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, backgroundColor: "#eee" },
  activeLangChip: { backgroundColor: "#007AFF" },
  langChipText: { fontSize: 12 },
  activeLangChipText: { color: "#fff", fontWeight: "bold" },
  chaptersList: { gap: 10 },
  chapterItem: { padding: 12, backgroundColor: "#fafafa", borderRadius: 8, borderWidth: 1, borderColor: "#f0f0f0" },
  chapterRow: { flexDirection: "row", justifyContent: "space-between" },
  chapterNumber: { fontSize: 16, fontWeight: "600", color: "#007AFF" },
  langTag: { fontSize: 10, color: "#999" },
  chapterTitle: { fontSize: 14, color: "#333", marginVertical: 2 },
  chapterPages: { fontSize: 12, color: "#999" },
  noChapters: { textAlign: "center", color: "#999", marginTop: 20 },
});