import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { BACKEND_URL } from "@/config";

export type ChapterNavEntry = {
  id: string;
  chapterLabel: string;
};

interface MangaReaderProps {
  pages: string[];
  prevChapter?: ChapterNavEntry | null;
  nextChapter?: ChapterNavEntry | null;
  onChapterNavigate?: (chapter: ChapterNavEntry) => void;
}

interface TranslationBubble {
  bubble_index: number;
  original_text: string;
  translated_text: string;
}

export let currentMangaPage = 1;

export default function MangaReader({
  pages,
  prevChapter = null,
  nextChapter = null,
  onChapterNavigate,
}: MangaReaderProps) {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();
  const listRef = useRef<FlatList<string>>(null);

  const [currentPage, setCurrentPage] = useState(1);
  const [autoTranslateEnabled, setAutoTranslateEnabled] = useState(true);
  const [translationsByPage, setTranslationsByPage] = useState<
    Record<number, TranslationBubble[]>
  >({});
  const [inFlightByPage, setInFlightByPage] = useState<Record<number, boolean>>(
    {},
  );
  const inFlightPagesRef = useRef<Set<number>>(new Set());
  const abortControllersRef = useRef<Map<number, AbortController>>(new Map());
  const LOOKAHEAD_PANELS = 3;
  const KEEP_PREVIOUS_PAGES = 1;
  const SCROLL_EVENT_THROTTLE = 100;

  const bottomInset = Math.max(insets.bottom, 8);
  const topInset = Math.max(insets.top, 8);
  const pageChromeBottom = bottomInset + 16;
  /** Browser vertical scrollbar overlaps the viewport on web; reserve space so chrome doesn't sit under it. */
  const webScrollbarGutter = Platform.OS === "web" ? 22 : 0;
  const translateToolbarRight =
    12 + Math.max(insets.right, 0) + webScrollbarGutter;

  const getItemLayout = (
    _data: ArrayLike<string> | null | undefined,
    index: number,
  ) => ({
    length: screenHeight,
    offset: screenHeight * index,
    index,
  });

  const isValidPage = (pageNumber: number) =>
    pageNumber >= 1 && pageNumber <= pages.length;

  const renderPage = ({ item, index }: { item: string; index: number }) => (
    <View
      style={[
        styles.pageContainer,
        { width: screenWidth, height: screenHeight },
      ]}
    >
      <Image source={{ uri: item }} style={styles.page} resizeMode="contain" />
      <Text style={[styles.pageNumber, { bottom: pageChromeBottom }]}>
        Page {index + 1}/{pages.length}
      </Text>
    </View>
  );

  const markPageInFlight = (pageNumber: number, inFlight: boolean) => {
    setInFlightByPage((prev) => ({
      ...prev,
      [pageNumber]: inFlight,
    }));
  };

  const getPageFromScroll = (offsetY: number, viewportHeight: number) => {
    const rawIndex = Math.round(offsetY / viewportHeight);
    return Math.max(0, Math.min(rawIndex, pages.length - 1)) + 1;
  };

  const requestPageTranslation = async (pageNumber: number) => {
    if (!isValidPage(pageNumber)) {
      return;
    }

    if (
      translationsByPage[pageNumber] ||
      inFlightPagesRef.current.has(pageNumber)
    ) {
      return;
    }

    inFlightPagesRef.current.add(pageNumber);
    markPageInFlight(pageNumber, true);
    const abortController = new AbortController();
    abortControllersRef.current.set(pageNumber, abortController);

    const backendUrl = `${BACKEND_URL}/api/manga/translate`;
    try {
      const res = await fetch(backendUrl, {
        method: "POST",
        body: JSON.stringify({
          image_url: pages[pageNumber - 1],
          language: "",
        }),
        headers: {
          "Content-Type": "application/json",
        },
        signal: abortController.signal,
      });
      const json = await res.json();

      if (json?.status === "success" && Array.isArray(json?.data)) {
        setTranslationsByPage((prev) => ({
          ...prev,
          [pageNumber]: json.data as TranslationBubble[],
        }));
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        /* cancelled */
      } else {
        console.error(`Failed to translate page ${pageNumber}`, err);
      }
    } finally {
      inFlightPagesRef.current.delete(pageNumber);
      abortControllersRef.current.delete(pageNumber);
      markPageInFlight(pageNumber, false);
    }
  };

  const handleTranslate = async () => {
    await requestPageTranslation(currentPage);
  };

  const prefetchLookaheadPages = (basePage: number) => {
    for (let i = 0; i <= LOOKAHEAD_PANELS; i++) {
      const pageToTranslate = basePage + i;
      if (pageToTranslate <= pages.length) {
        void requestPageTranslation(pageToTranslate);
      }
    }
  };

  const cancelStaleInFlightRequests = (basePage: number) => {
    const minPageToKeep = Math.max(1, basePage - KEEP_PREVIOUS_PAGES);
    const maxPageToKeep = Math.min(pages.length, basePage + LOOKAHEAD_PANELS);

    for (const page of inFlightPagesRef.current) {
      const outsideWindow = page < minPageToKeep || page > maxPageToKeep;
      if (outsideWindow) {
        abortControllersRef.current.get(page)?.abort();
      }
    }
  };

  useEffect(() => {
    if (!autoTranslateEnabled || pages.length === 0) {
      return;
    }
    cancelStaleInFlightRequests(currentPage);
    prefetchLookaheadPages(currentPage);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- helpers close over latest pages each render
  }, [currentPage, autoTranslateEnabled, pages.length]);

  if (pages.length === 0) {
    return (
      <View style={[styles.pageContainer, styles.centered]}>
        <Text>No pages available</Text>
      </View>
    );
  }

  const handleScroll = (offsetY: number, viewportHeight: number) => {
    const nextPage = getPageFromScroll(offsetY, viewportHeight);
    currentMangaPage = nextPage;

    if (nextPage !== currentPage) {
      setCurrentPage(nextPage);
    }
  };

  const showChapterFooter =
    onChapterNavigate != null && (prevChapter != null || nextChapter != null);

  const chapterFooter = showChapterFooter ? (
    <View
      style={[styles.chapterFooterChrome, { paddingBottom: bottomInset + 16 }]}
    >
      <View style={styles.chapterFooterRow}>
        {prevChapter != null ? (
          <Pressable
            style={({ pressed }) => [
              styles.chromeBtn,
              pressed && styles.chromeBtnPressed,
            ]}
            onPress={() => onChapterNavigate(prevChapter)}
            accessibilityRole="button"
            accessibilityLabel="Previous chapter"
          >
            <Ionicons name="chevron-back" size={26} color="#fff" />
            <View style={styles.chapterFooterBtnText}>
              <Text style={styles.chromeBtnLabel}>Prev chapter</Text>
              <Text style={styles.chapterFooterSub}>
                Ch. {prevChapter.chapterLabel}
              </Text>
            </View>
          </Pressable>
        ) : (
          <View style={styles.chapterFooterSpacer} />
        )}
        {nextChapter != null ? (
          <Pressable
            style={({ pressed }) => [
              styles.chromeBtn,
              styles.chapterFooterNext,
              pressed && styles.chromeBtnPressed,
            ]}
            onPress={() => onChapterNavigate(nextChapter)}
            accessibilityRole="button"
            accessibilityLabel="Next chapter"
          >
            <View style={styles.chapterFooterBtnText}>
              <Text style={styles.chromeBtnLabel}>Next chapter</Text>
              <Text style={styles.chapterFooterSub}>
                Ch. {nextChapter.chapterLabel}
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={26} color="#fff" />
          </Pressable>
        ) : (
          <View style={styles.chapterFooterSpacer} />
        )}
      </View>
    </View>
  ) : null;

  return (
    <View style={{ flex: 1 }}>
      <Pressable
        style={({ pressed }) => [
          styles.topReturnBtn,
          { top: topInset + 4, left: 12 },
          pressed && styles.chromeBtnPressed,
        ]}
        onPress={() => router.back()}
        accessibilityRole="button"
        accessibilityLabel="Go back"
        hitSlop={8}
      >
        <Ionicons name="arrow-back" size={22} color="#fff" />
        <Text style={styles.topReturnLabel}>Return</Text>
      </Pressable>

      <View
        style={[
          styles.translateToolbar,
          {
            top: topInset + 4,
            right: translateToolbarRight,
            maxWidth: Math.min(screenWidth * 0.52, 268),
          },
        ]}
        pointerEvents="box-none"
      >
        <TouchableOpacity
          style={styles.menuButton}
          onPress={() => handleTranslate()}
        >
          <Text style={styles.menuButtonText}>Translate</Text>
        </TouchableOpacity>
        <View style={styles.autoTranslateToggleRow}>
          <Text style={styles.autoTranslateToggleText}>Auto translate</Text>
          <Switch
            value={autoTranslateEnabled}
            onValueChange={setAutoTranslateEnabled}
            thumbColor="#fff"
            trackColor={{
              false: "rgba(255,255,255,0.25)",
              true: "rgba(34,197,94,0.7)",
            }}
          />
        </View>
        {inFlightByPage[currentPage] && (
          <View style={styles.inFlightIndicator}>
            <ActivityIndicator size="small" color="#fff" />
            <Text style={styles.inFlightText}>Translating current page...</Text>
          </View>
        )}

        {translationsByPage[currentPage] && (
          <View style={styles.translationPanel}>
            <Text style={styles.translationTitle}>
              Page {currentPage} translations
            </Text>
            {translationsByPage[currentPage].map((bubble) => (
              <View key={bubble.bubble_index} style={styles.translationItem}>
                <Text style={styles.translationText}>
                  {bubble.original_text}
                </Text>
                <Text style={styles.translationText}>
                  {bubble.translated_text}
                </Text>
              </View>
            ))}
          </View>
        )}
      </View>

      <View style={{ flex: 1 }}>
        <FlatList
          ref={listRef}
          data={pages}
          keyExtractor={(item, index) => index.toString()}
          renderItem={renderPage}
          initialNumToRender={3}
          maxToRenderPerBatch={10}
          windowSize={100}
          getItemLayout={getItemLayout}
          onScrollToIndexFailed={(info) => {
            const offset = info.index * screenHeight;
            listRef.current?.scrollToOffset({ offset, animated: true });
          }}
          onScroll={(e) => {
            const offsetY = e.nativeEvent.contentOffset.y;
            const viewportHeight =
              e.nativeEvent.layoutMeasurement.height || screenHeight;
            handleScroll(offsetY, viewportHeight);
          }}
          scrollEventThrottle={SCROLL_EVENT_THROTTLE}
          ListFooterComponent={chapterFooter}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  pageContainer: {
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  page: {
    width: "100%",
    height: "100%",
  },
  pageNumber: {
    position: "absolute",
    alignSelf: "center",
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
    backgroundColor: "rgba(0, 0, 0, 0.4)",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    overflow: "hidden",
    zIndex: 10,
  },
  translateToolbar: {
    position: "absolute",
    zIndex: 38,
    alignItems: "stretch",
    alignSelf: "flex-end",
  },
  menuButton: {
    alignSelf: "flex-end",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
  },
  menuButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
  autoTranslateToggleRow: {
    marginTop: 8,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
    width: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  autoTranslateToggleText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "500",
  },
  inFlightIndicator: {
    marginTop: 8,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    alignSelf: "stretch",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  inFlightText: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "500",
  },
  translationPanel: {
    marginTop: 10,
    alignSelf: "stretch",
    maxHeight: 250,
    backgroundColor: "rgba(0, 0, 0, 0.75)",
    borderRadius: 10,
    padding: 10,
  },
  translationTitle: {
    color: "#fff",
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 8,
  },
  translationItem: {
    marginBottom: 8,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255, 255, 255, 0.2)",
  },
  translationText: {
    color: "#fff",
    fontSize: 12,
    lineHeight: 18,
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  topReturnBtn: {
    position: "absolute",
    zIndex: 40,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: "rgba(17, 24, 39, 0.85)",
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(255,255,255,0.15)",
  },
  topReturnLabel: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "700",
  },
  chapterFooterChrome: {
    backgroundColor: "rgba(17, 24, 39, 0.96)",
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(255,255,255,0.12)",
    paddingTop: 14,
    paddingHorizontal: 16,
    minHeight: 96,
    justifyContent: "center",
  },
  chapterFooterRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  chapterFooterSpacer: {
    flex: 1,
    minWidth: 88,
  },
  chapterFooterNext: {
    flexDirection: "row-reverse",
  },
  chapterFooterBtnText: {
    alignItems: "flex-start",
    gap: 2,
    maxWidth: 140,
  },
  chapterFooterSub: {
    color: "rgba(255,255,255,0.65)",
    fontSize: 11,
    fontWeight: "600",
  },
  chromeBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    minWidth: 88,
    flex: 1,
    justifyContent: "center",
  },
  chromeBtnPressed: {
    opacity: 0.82,
  },
  chromeBtnLabel: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "700",
  },
});
