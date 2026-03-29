import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import { useAuth } from "@/context/AuthContext";
import {
  fetchReadingListItems,
  removeReadingListItem,
  type ReadingListItem,
} from "@/lib/readingListApi";
import { fetchMangaCoverUrl } from "@/lib/mangaCoverApi";
import PopUp from "@/app/components/PopUp";
import { BACKEND_URL } from "@/app/config";
import type { Chapter, Manga } from "@/app/types/types";

const COVER_PLACEHOLDER =
  "https://via.placeholder.com/128x180?text=No+Cover";

const cardShadow = Platform.select({
  ios: {
    shadowColor: "#0f172a",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
  },
  android: { elevation: 3 },
  default: {},
});

function normalizeParam(v: string | string[] | undefined): string | undefined {
  if (v == null) return undefined;
  return Array.isArray(v) ? v[0] : v;
}

function mapMdJsonToManga(data: Record<string, unknown>): Manga | null {
  if (typeof data.id !== "string") return null;
  const attrs = data.attributes as Record<string, unknown> | undefined;
  const titleRaw = attrs?.title as Record<string, string> | undefined;
  const descRaw = attrs?.description as Record<string, string> | undefined;
  const langs = attrs?.availableTranslatedLanguages as string[] | undefined;
  const rels = (data.relationships as unknown[]) ?? [];
  return {
    id: data.id,
    attributes: {
      title:
        titleRaw && typeof titleRaw === "object" ? titleRaw : { en: "Untitled" },
      description:
        descRaw && typeof descRaw === "object"
          ? { en: descRaw.en ?? descRaw["en"] }
          : {},
      availableTranslatedLanguages: Array.isArray(langs) ? langs : [],
    },
    relationships: rels.map((r: unknown) => {
      const o = r as Record<string, unknown>;
      return {
        type: String(o.type ?? ""),
        attributes: o.attributes as { fileName?: string } | undefined,
      };
    }),
  };
}

async function fetchMangaDexManga(mangaId: string): Promise<Manga | null> {
  try {
    const res = await fetch(
      `https://api.mangadex.org/manga/${mangaId}?includes[]=cover_art`,
    );
    if (!res.ok) return null;
    const json = (await res.json()) as { data?: Record<string, unknown> };
    const data = json.data;
    if (!data) return null;
    return mapMdJsonToManga(data);
  } catch {
    return null;
  }
}

function coverUrlFromManga(manga: Manga): string {
  const coverArt = manga.relationships?.find((rel) => rel.type === "cover_art");
  const fileName = coverArt?.attributes?.fileName;
  return fileName
    ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
    : COVER_PLACEHOLDER;
}

async function fetchChaptersForManga(mangaId: string): Promise<Chapter[]> {
  const res = await fetch(`${BACKEND_URL}/api/manga/${mangaId}/chapters`);
  if (!res.ok) return [];
  const json = await res.json();
  const rows = (json.data || []) as Record<string, unknown>[];
  return rows.map((ch) => {
    const a = (ch.attributes ?? {}) as Record<string, unknown>;
    return {
      id: String(ch.id),
      chapter: String(a.chapter ?? "?"),
      title: String(a.title ?? ""),
      pages: typeof a.pages === "number" ? a.pages : 0,
      language: String(a.translatedLanguage ?? "unknown"),
    };
  });
}

async function hydrateMangaCovers(
  items: ReadingListItem[],
  setCoverUrls: React.Dispatch<
    React.SetStateAction<Record<string, string | null>>
  >,
) {
  const extIds = [
    ...new Set(items.map((i) => i.external_manga_id).filter(Boolean)),
  ] as string[];
  setCoverUrls((prev) => {
    const need = extIds.filter((id) => !(id in prev));
    if (need.length === 0) return prev;
    void (async () => {
      const results = Object.fromEntries(
        await Promise.all(
          need.map(async (id) => [id, await fetchMangaCoverUrl(id)] as const),
        ),
      ) as Record<string, string | null>;
      setCoverUrls((p) => ({ ...p, ...results }));
    })();
    return prev;
  });
}

export default function ReadingListDetailScreen() {
  const router = useRouter();
  const { id: rawId, title: rawTitle } = useLocalSearchParams<{
    id: string | string[];
    title?: string | string[];
  }>();
  const { session, loading: authLoading } = useAuth();

  const idStr = normalizeParam(rawId);
  const listId = idStr != null ? Number(idStr) : NaN;
  const listTitle =
    normalizeParam(rawTitle) ?? "Reading list";

  const [items, setItems] = useState<ReadingListItem[] | null>(null);
  const [coverUrls, setCoverUrls] = useState<Record<string, string | null>>({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [removingKey, setRemovingKey] = useState<string | null>(null);

  const [popupVisible, setPopupVisible] = useState(false);
  const [popupManga, setPopupManga] = useState<Manga | null>(null);
  const [popupChapters, setPopupChapters] = useState<Chapter[]>([]);
  const [loadingPopupChapters, setLoadingPopupChapters] = useState(false);
  const [openingPopup, setOpeningPopup] = useState(false);

  const loadItems = useCallback(async () => {
    if (!session?.access_token || !Number.isFinite(listId)) return;
    setError(null);
    try {
      const data = await fetchReadingListItems(session.access_token, listId);
      setItems(data);
      await hydrateMangaCovers(data, setCoverUrls);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not load this reading list.",
      );
      setItems([]);
    }
  }, [session?.access_token, listId]);

  useEffect(() => {
    if (authLoading) return;
    if (!session) {
      router.replace("/sign-in");
      return;
    }
    if (!session.access_token) return;
    if (!Number.isFinite(listId)) {
      setLoading(false);
      setError("Invalid list.");
      return;
    }
    setLoading(true);
    void loadItems().finally(() => setLoading(false));
  }, [authLoading, session, listId, loadItems, router]);

  const onRefresh = useCallback(async () => {
    if (!session?.access_token) return;
    setRefreshing(true);
    try {
      await loadItems();
    } finally {
      setRefreshing(false);
    }
  }, [session?.access_token, loadItems]);

  const openPopupForItem = async (item: ReadingListItem) => {
    const extId = item.external_manga_id;
    if (extId == null || extId === "") {
      Alert.alert(
        "Unavailable",
        "This entry has no linked MangaDex id, so details can't be opened.",
      );
      return;
    }
    setOpeningPopup(true);
    setPopupChapters([]);
    setLoadingPopupChapters(false);
    try {
      const manga = await fetchMangaDexManga(extId);
      if (!manga) {
        Alert.alert(
          "Couldn't load manga",
          "This title may be unavailable on MangaDex.",
        );
        return;
      }
      setPopupManga(manga);
      setPopupVisible(true);
      setLoadingPopupChapters(true);
      try {
        const ch = await fetchChaptersForManga(extId);
        setPopupChapters(ch);
      } catch {
        setPopupChapters([]);
      } finally {
        setLoadingPopupChapters(false);
      }
    } finally {
      setOpeningPopup(false);
    }
  };

  const closePopup = () => {
    setPopupVisible(false);
    setPopupManga(null);
    setPopupChapters([]);
  };

  const onRemoveItem = async (mangaId: number) => {
    if (!session?.access_token || !Number.isFinite(listId)) return;
    const key = `${listId}-${mangaId}`;
    setRemovingKey(key);
    setError(null);
    try {
      await removeReadingListItem(session.access_token, listId, mangaId);
      setItems((prev) =>
        prev ? prev.filter((x) => x.manga_id !== mangaId) : prev,
      );
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not remove from list.",
      );
    } finally {
      setRemovingKey(null);
    }
  };

  if (authLoading || !session) {
    return (
      <SafeAreaView style={styles.safe}>
        <ActivityIndicator style={styles.loader} color="#374151" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={["top", "left", "right"]}>
      <View style={styles.pageCenter}>
        <View style={styles.toolbar}>
          <Pressable
            onPress={() => router.back()}
            style={({ pressed }) => [
              styles.backBtn,
              pressed && styles.backBtnPressed,
            ]}
          >
            <Text style={styles.backBtnText}>← Back</Text>
          </Pressable>
        </View>

        <ScrollView
          contentContainerStyle={styles.scroll}
          keyboardShouldPersistTaps="handled"
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor="#374151"
              colors={["#111827"]}
            />
          }
        >
          <Text style={styles.pageTitle} numberOfLines={2}>
            {listTitle}
          </Text>
          <Text style={styles.subtitle}>
            {items == null && loading
              ? "Loading…"
              : `${items?.length ?? 0} title${
                  items?.length === 1 ? "" : "s"
                }`}
          </Text>

          {error ? (
            <View style={styles.noticeError}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          {loading && items == null ? (
            <ActivityIndicator
              style={styles.spinner}
              color="#374151"
              size="large"
            />
          ) : null}

          {!loading && items && items.length === 0 ? (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyText}>
                No manga in this list yet. Add some from search or the home
                screen.
              </Text>
            </View>
          ) : null}

          {items?.map((item) => {
            const rk = `${listId}-${item.manga_id}`;
            const extId = item.external_manga_id;
            const resolvedCover =
              extId != null && extId !== "" ? coverUrls[extId] : null;
            const coverUri = resolvedCover ?? COVER_PLACEHOLDER;
            const canOpen = extId != null && extId !== "";
            return (
              <View key={item.id} style={styles.listRow}>
                <Pressable
                  style={({ pressed }) => [
                    styles.rowMainPress,
                    pressed && canOpen && styles.rowMainPressed,
                    !canOpen && styles.rowMainDisabled,
                  ]}
                  onPress={() => void openPopupForItem(item)}
                  disabled={openingPopup}
                >
                  <Image
                    source={{ uri: coverUri }}
                    style={styles.thumb}
                    resizeMode="cover"
                  />
                  <View style={styles.rowText}>
                    <Text style={styles.rowTitle} numberOfLines={2}>
                      {item.manga_title}
                    </Text>
                    {item.last_chapter_number != null ? (
                      <Text style={styles.rowMeta}>
                        Last read · Ch. {item.last_chapter_number}
                      </Text>
                    ) : null}
                    {canOpen ? (
                      <Text style={styles.tapHint}>
                        Tap for details and chapters
                      </Text>
                    ) : null}
                  </View>
                </Pressable>
                <Pressable
                  style={({ pressed }) => [
                    styles.removeBtn,
                    pressed && styles.removeBtnPressed,
                  ]}
                  onPress={() => void onRemoveItem(item.manga_id)}
                  disabled={removingKey === rk}
                >
                  {removingKey === rk ? (
                    <ActivityIndicator size="small" color="#b91c1c" />
                  ) : (
                    <Text style={styles.removeBtnText}>Remove</Text>
                  )}
                </Pressable>
              </View>
            );
          })}
        </ScrollView>

        {openingPopup ? (
          <View style={styles.popupLoadingOverlay} pointerEvents="none">
            <ActivityIndicator size="large" color="#374151" />
          </View>
        ) : null}
      </View>

      {popupManga ? (
        <PopUp
          visible={popupVisible}
          title={
            Object.values(popupManga.attributes.title)[0] || "Untitled"
          }
          summary={
            popupManga.attributes.description?.en || "No description available."
          }
          coverArt={coverUrlFromManga(popupManga)}
          manga={popupManga}
          chapters={popupChapters}
          loadingChapters={loadingPopupChapters}
          onClose={closePopup}
        />
      ) : null}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#f3f4f6" },
  loader: { marginTop: 48 },
  pageCenter: {
    flex: 1,
    width: "100%",
    maxWidth: 440,
    alignSelf: "center",
  },
  toolbar: {
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 4,
  },
  backBtn: {
    alignSelf: "flex-start",
    paddingVertical: 8,
    paddingHorizontal: 4,
  },
  backBtnPressed: { opacity: 0.55 },
  backBtnText: {
    fontSize: 16,
    color: "#2563eb",
    fontWeight: "600",
  },
  scroll: {
    paddingHorizontal: 20,
    paddingBottom: 32,
    alignItems: "center",
  },
  pageTitle: {
    fontSize: 26,
    fontWeight: "800",
    color: "#111827",
    textAlign: "center",
    letterSpacing: -0.4,
    marginBottom: 6,
    maxWidth: "100%",
  },
  subtitle: {
    fontSize: 14,
    color: "#6b7280",
    textAlign: "center",
    marginBottom: 20,
  },
  noticeError: {
    backgroundColor: "#fef2f2",
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#fecaca",
    width: "100%",
  },
  errorText: { color: "#b91c1c", fontSize: 14 },
  spinner: { marginVertical: 32 },
  emptyCard: {
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 22,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderStyle: "dashed",
    width: "100%",
    ...cardShadow,
  },
  emptyText: {
    fontSize: 15,
    color: "#6b7280",
    textAlign: "center",
    lineHeight: 22,
  },
  listRow: {
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    paddingVertical: 8,
    paddingRight: 8,
    paddingLeft: 4,
    marginBottom: 10,
    backgroundColor: "#fff",
    borderRadius: 14,
    gap: 8,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...cardShadow,
  },
  rowMainPress: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    minWidth: 0,
    paddingVertical: 4,
    paddingLeft: 8,
    borderRadius: 12,
  },
  rowMainPressed: { opacity: 0.88 },
  rowMainDisabled: { opacity: 0.72 },
  tapHint: {
    fontSize: 11,
    color: "#2563eb",
    fontWeight: "600",
    marginTop: 4,
  },
  popupLoadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(255,255,255,0.7)",
    justifyContent: "center",
    alignItems: "center",
  },
  thumb: {
    width: 52,
    height: 74,
    borderRadius: 8,
    backgroundColor: "#e5e7eb",
  },
  rowText: { flex: 1, minWidth: 0 },
  rowTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#111827",
  },
  rowMeta: { fontSize: 12, color: "#6b7280", marginTop: 4 },
  removeBtn: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    minWidth: 76,
    borderRadius: 10,
    backgroundColor: "#fef2f2",
  },
  removeBtnPressed: { backgroundColor: "#fee2e2" },
  removeBtnText: {
    fontSize: 13,
    color: "#b91c1c",
    fontWeight: "700",
    textAlign: "center",
  },
});
