import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { type Href, useLocalSearchParams, useRouter } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import PopUp from "@/app/components/PopUp";
import { useAuth } from "@/context/AuthContext";
import type { Chapter, Manga } from "@/app/types/types";
import {
  fetchReadingListItems,
  removeReadingListItem,
  type ReadingListItem,
} from "@/lib/readingListApi";
import {
  buildMangaForPopup,
  COVER_PLACEHOLDER,
  descriptionPreviewForTile,
  fetchChaptersForManga,
  hydrateMangaCovers,
  hydrateMangaInfos,
  pickChapterForListItem,
  titleForListItem,
  type MangaPublicInfo,
} from "@/lib/readingListDetailManga";
import ReadingListGridTile from "./ReadingListGridTile";

const gridGap = 8;

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

function normalizeRouteParam(
  v: string | string[] | undefined,
): string | undefined {
  if (v == null) return undefined;
  return Array.isArray(v) ? v[0] : v;
}

export default function ReadingListDetailScreen() {
  const router = useRouter();
  const { id: rawId, title: rawTitle } = useLocalSearchParams<{
    id: string | string[];
    title?: string | string[];
  }>();
  const { session, loading: authLoading } = useAuth();

  const idStr = normalizeRouteParam(rawId);
  const listId = idStr != null ? Number(idStr) : NaN;
  const listTitle = normalizeRouteParam(rawTitle) ?? "Reading list";

  const [items, setItems] = useState<ReadingListItem[] | null>(null);
  const [coverUrls, setCoverUrls] = useState<Record<string, string | null>>({});
  const [mangaInfos, setMangaInfos] = useState<
    Record<string, MangaPublicInfo | undefined>
  >({});

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [removingKey, setRemovingKey] = useState<string | null>(null);
  const [openingReaderKey, setOpeningReaderKey] = useState<string | null>(null);

  const [popupVisible, setPopupVisible] = useState(false);
  const [popupManga, setPopupManga] = useState<Manga | null>(null);
  const [popupCoverArt, setPopupCoverArt] = useState(COVER_PLACEHOLDER);
  const [popupChapters, setPopupChapters] = useState<Chapter[]>([]);
  const [popupChaptersLoading, setPopupChaptersLoading] = useState(false);

  const loadItems = useCallback(async () => {
    if (!session?.access_token || !Number.isFinite(listId)) return;
    setError(null);
    try {
      const data = await fetchReadingListItems(session.access_token, listId);
      setItems(data);
      setMangaInfos({});
      void hydrateMangaCovers(data, setCoverUrls);
      void hydrateMangaInfos(data, setMangaInfos);
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

  const openReaderForItem = async (item: ReadingListItem, rowKey: string) => {
    const extId = item.external_manga_id;
    if (extId == null || extId === "") {
      Alert.alert(
        "Unavailable",
        "This entry has no linked MangaDex id, so the reader can't be opened.",
      );
      return;
    }
    setOpeningReaderKey(rowKey);
    try {
      const chapters = await fetchChaptersForManga(extId);
      const chapter = pickChapterForListItem(
        chapters,
        item.last_chapter_number,
      );
      if (!chapter) {
        Alert.alert("No chapters", "No chapters were found for this title.");
        return;
      }
      router.push(`/reader/${chapter.id}` as Href);
    } catch {
      Alert.alert("Error", "Could not load chapters for this title.");
    } finally {
      setOpeningReaderKey(null);
    }
  };

  const openDetailPopup = useCallback(
    (item: ReadingListItem, extId: string) => {
      const info = mangaInfos[extId];
      const cov = coverUrls[extId] ?? null;
      setPopupManga(buildMangaForPopup(extId, item, info, cov));
      setPopupCoverArt(cov ?? COVER_PLACEHOLDER);
      setPopupVisible(true);
      setPopupChapters([]);
      setPopupChaptersLoading(true);
      void fetchChaptersForManga(extId)
        .then(setPopupChapters)
        .finally(() => setPopupChaptersLoading(false));
    },
    [mangaInfos, coverUrls],
  );

  const closeDetailPopup = useCallback(() => {
    setPopupVisible(false);
    setPopupManga(null);
    setPopupChapters([]);
    setPopupChaptersLoading(false);
    setPopupCoverArt(COVER_PLACEHOLDER);
  }, []);

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

  const readerBusy = openingReaderKey !== null;

  if (authLoading || !session) {
    return (
      <SafeAreaView style={styles.safe} edges={["top", "right", "bottom"]}>
        <ActivityIndicator style={styles.loader} color="#374151" />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={["top", "right", "bottom"]}>
      <View style={styles.page}>
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
          <View style={styles.column}>
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

            {items != null && items.length > 0 ? (
              <View style={[styles.gridWrap, { gap: gridGap }]}>
                {items.map((item) => {
                  const rk = `${listId}-${item.manga_id}`;
                  const extId = item.external_manga_id;
                  const resolved =
                    extId != null && extId !== ""
                      ? coverUrls[extId]
                      : null;
                  const coverUri = resolved ?? COVER_PLACEHOLDER;
                  const canOpen = extId != null && extId !== "";
                  const info = extId ? mangaInfos[extId] : undefined;

                  const { descLoading, previewText, showReadMore } =
                    descriptionPreviewForTile(extId, info);

                  return (
                    <ReadingListGridTile
                      key={item.id}
                      item={item}
                      coverUri={coverUri}
                      displayTitle={titleForListItem(item, info)}
                      descLoading={descLoading}
                      previewText={previewText}
                      showReadMore={showReadMore}
                      extId={extId}
                      canOpen={canOpen}
                      openingThisTile={openingReaderKey === rk}
                      readerBusy={readerBusy}
                      removing={removingKey === rk}
                      onCoverPress={() => void openReaderForItem(item, rk)}
                      onChapterPress={() => void openReaderForItem(item, rk)}
                      onReadMore={() => {
                        if (extId != null) openDetailPopup(item, extId);
                      }}
                      onRemove={() => void onRemoveItem(item.manga_id)}
                    />
                  );
                })}
              </View>
            ) : null}
          </View>
        </ScrollView>
      </View>

      {popupManga ? (
        <PopUp
          visible={popupVisible}
          title={
            Object.values(popupManga.attributes.title)[0] || "Untitled"
          }
          summary={
            popupManga.attributes.description?.en ||
            "No description available."
          }
          coverArt={popupCoverArt}
          manga={popupManga}
          chapters={popupChapters}
          loadingChapters={popupChaptersLoading}
          onClose={closeDetailPopup}
        />
      ) : null}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#f3f4f6" },
  loader: { marginTop: 48 },
  page: {
    flex: 1,
    width: "100%",
    alignItems: "stretch",
  },
  toolbar: {
    paddingHorizontal: 20,
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
    flexGrow: 1,
    width: "100%",
    paddingVertical: 24,
    paddingBottom: 40,
    paddingHorizontal: 20,
    alignItems: "stretch",
  },
  column: {
    width: "100%",
    alignItems: "stretch",
  },
  pageTitle: {
    fontSize: 28,
    fontWeight: "800",
    color: "#111827",
    textAlign: "left",
    letterSpacing: -0.4,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: "#6b7280",
    textAlign: "left",
    lineHeight: 20,
    marginBottom: 16,
  },
  noticeError: {
    backgroundColor: "#fef2f2",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#fecaca",
    width: "100%",
  },
  errorText: {
    color: "#b91c1c",
    fontSize: 14,
    textAlign: "left",
  },
  spinner: { marginVertical: 24 },
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
    textAlign: "left",
    lineHeight: 22,
  },
  gridWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    width: "100%",
    marginBottom: 12,
    alignItems: "flex-start",
  },
});
