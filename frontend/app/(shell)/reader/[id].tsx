import { useLocalSearchParams, useRouter, type Href } from "expo-router";
import { Asset } from "expo-asset";
import { useCallback, useEffect, useState } from "react";
import { ActivityIndicator, View } from "react-native";
import MangaReader, {
  type ChapterNavEntry,
} from "@/app/components/MangaReader";

import { BACKEND_URL } from "@/config";
import { useAuth } from "@/context/AuthContext";
import {
  fetchChaptersForManga,
  parseChapterNumber,
  sortChaptersAscending,
} from "@/lib/readingListDetailManga";
import { patchReadingListItemProgress } from "@/lib/readingListApi";

const IS_TESTING = false; // true for testing with local images

function oneParam(v: string | string[] | undefined): string | undefined {
  if (v == null) return undefined;
  return Array.isArray(v) ? v[0] : v;
}

function buildReaderHref(
  chapterId: string,
  opts: {
    seriesId?: string;
    readingListId?: string;
    mangaId?: string;
    chapterNumber?: number | null;
  },
): Href {
  const q = new URLSearchParams();
  if (opts.seriesId != null && opts.seriesId !== "") {
    q.set("seriesId", opts.seriesId);
  }
  if (opts.readingListId != null && opts.readingListId !== "") {
    q.set("readingListId", opts.readingListId);
  }
  if (opts.mangaId != null && opts.mangaId !== "") {
    q.set("mangaId", opts.mangaId);
  }
  if (opts.chapterNumber != null && Number.isFinite(opts.chapterNumber)) {
    q.set("chapterNumber", String(opts.chapterNumber));
  }
  const qs = q.toString();
  return (qs ? `/reader/${chapterId}?${qs}` : `/reader/${chapterId}`) as Href;
}

export default function ReaderScreen() {
  const params = useLocalSearchParams();
  const router = useRouter();
  const chapterMdId = oneParam(params.id as string | string[] | undefined);
  const seriesIdStr = oneParam(
    params.seriesId as string | string[] | undefined,
  );
  const readingListIdStr = oneParam(
    params.readingListId as string | string[] | undefined,
  );
  const mangaIdStr = oneParam(params.mangaId as string | string[] | undefined);
  const chapterNumberStr = oneParam(
    params.chapterNumber as string | string[] | undefined,
  );

  const { session } = useAuth();
  const [pages, setPages] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [prevChapter, setPrevChapter] = useState<ChapterNavEntry | null>(null);
  const [nextChapter, setNextChapter] = useState<ChapterNavEntry | null>(null);

  useEffect(() => {
    const rl = readingListIdStr != null ? Number(readingListIdStr) : NaN;
    const mid = mangaIdStr != null ? Number(mangaIdStr) : NaN;
    const ch = chapterNumberStr != null ? Number(chapterNumberStr) : NaN;
    if (
      !session?.access_token ||
      !Number.isFinite(rl) ||
      !Number.isFinite(mid) ||
      !Number.isFinite(ch)
    ) {
      return;
    }
    void patchReadingListItemProgress(
      session.access_token,
      rl,
      mid,
      ch,
    ).catch(() => {});
  }, [
    session?.access_token,
    readingListIdStr,
    mangaIdStr,
    chapterNumberStr,
  ]);

  useEffect(() => {
    if (!chapterMdId || !seriesIdStr) {
      setPrevChapter(null);
      setNextChapter(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      const raw = await fetchChaptersForManga(seriesIdStr);
      const chapters = sortChaptersAscending(raw);
      const idx = chapters.findIndex((c) => c.id === chapterMdId);
      if (cancelled) return;
      if (idx < 0) {
        setPrevChapter(null);
        setNextChapter(null);
        return;
      }
      const prev = idx > 0 ? chapters[idx - 1] : null;
      const next = idx < chapters.length - 1 ? chapters[idx + 1] : null;
      setPrevChapter(
        prev
          ? { id: prev.id, chapterLabel: prev.chapter }
          : null,
      );
      setNextChapter(
        next
          ? { id: next.id, chapterLabel: next.chapter }
          : null,
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [chapterMdId, seriesIdStr]);

  const onChapterNavigate = useCallback(
    (entry: ChapterNavEntry) => {
      const chNum = parseChapterNumber(entry.chapterLabel);
      router.replace(
        buildReaderHref(entry.id, {
          seriesId: seriesIdStr,
          readingListId: readingListIdStr,
          mangaId: mangaIdStr,
          chapterNumber: chNum,
        }),
      );
    },
    [router, seriesIdStr, readingListIdStr, mangaIdStr],
  );

  const runTest = async () => {
    const testPages = [
      Asset.fromModule(require("../../../assets/images/test_1.png")).uri,
      Asset.fromModule(require("../../../assets/images/test_7.png")).uri,
      Asset.fromModule(require("../../../assets/images/cntest_1.png")).uri,
      Asset.fromModule(require("../../../assets/images/krtest_1.png")).uri,
    ];
    setPages(testPages);
  };

  useEffect(() => {
    const fetchPages = async () => {
      setLoading(true);
      try {
        if (IS_TESTING) {
          await runTest();
        } else {
          try {
            const res = await fetch(
              `${BACKEND_URL}/api/manga/chapter/${chapterMdId}/pages`,
            );

            if (!res.ok) {
              throw new Error(`Server responded with ${res.status}`);
            }

            const json = await res.json();
            if (json.urls && Array.isArray(json.urls)) {
              setPages(json.urls);
            } else {
              console.warn("Backend returned no URLs for this chapter.");
              setPages([]);
            }
          } catch (error) {
            console.error("Error fetching chapter pages:", error);
            setPages([]);
          }
        }
      } catch (error) {
        console.error("Failed to fetch chapter pages:", error);
      } finally {
        setLoading(false);
      }
    };

    if (chapterMdId) {
      void fetchPages();
    }
  }, [chapterMdId]);

  if (loading) {
    return (
      <View
        style={{ flex: 1, justifyContent: "center", alignItems: "center" }}
      >
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  return (
    <MangaReader
      pages={pages}
      prevChapter={prevChapter}
      nextChapter={nextChapter}
      onChapterNavigate={
        seriesIdStr ? onChapterNavigate : undefined
      }
    />
  );
}
