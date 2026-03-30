/**
 * MangaDex fetch + pure helpers for the reading-list detail screen.
 */
import type { Dispatch, SetStateAction } from "react";
import { BACKEND_URL } from "@/app/config";
import type { Chapter, Manga } from "@/app/types/types";
import { fetchMangaCoverUrl } from "@/lib/mangaCoverApi";
import type { ReadingListItem } from "@/lib/readingListApi";

export const COVER_PLACEHOLDER =
  "https://via.placeholder.com/200x300?text=No+Cover";

/** Synopsis length on grid tiles before showing “Read more”. */
export const DESC_PREVIEW_CHARS = 120;

export function titleForListItem(
  item: ReadingListItem,
  info: MangaPublicInfo | undefined,
): string {
  return info?.title?.trim() || item.manga_title;
}

export function descriptionPreviewForTile(
  extId: string | null,
  info: MangaPublicInfo | undefined,
): {
  descLoading: boolean;
  previewText: string;
  showReadMore: boolean;
} {
  const descLoading = Boolean(extId && info === undefined);
  const full = !extId
    ? "No MangaDex link — description unavailable."
    : descLoading
      ? ""
      : info?.description?.trim() || "No description.";
  const showReadMore =
    Boolean(extId) &&
    !descLoading &&
    full.length > DESC_PREVIEW_CHARS;
  const previewText = showReadMore
    ? `${full.slice(0, DESC_PREVIEW_CHARS).trimEnd()}…`
    : full;
  return { descLoading, previewText, showReadMore };
}

export type MangaPublicInfo = {
  title: string | null;
  description: string | null;
  availableTranslatedLanguages: string[];
};

export function parseChapterNumber(chapterLabel: string): number | null {
  const s = chapterLabel.trim();
  if (s === "" || s === "?") return null;
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : null;
}

export function sortChaptersAscending(chapters: Chapter[]): Chapter[] {
  return [...chapters].sort((a, b) => {
    const na = parseChapterNumber(a.chapter);
    const nb = parseChapterNumber(b.chapter);
    if (na !== null && nb !== null) return na - nb;
    if (na !== null) return -1;
    if (nb !== null) return 1;
    return 0;
  });
}

/** Last-read chapter, or nearest earlier, else first (ascending order). */
export function pickChapterForListItem(
  chapters: Chapter[],
  lastRead: number | null | undefined,
): Chapter | null {
  const sorted = sortChaptersAscending(chapters);
  if (sorted.length === 0) return null;
  if (lastRead == null || !Number.isFinite(lastRead)) {
    return sorted[0];
  }
  const eps = 1e-4;
  const exact = sorted.find((c) => {
    const n = parseChapterNumber(c.chapter);
    return n !== null && Math.abs(n - lastRead) < eps;
  });
  if (exact) return exact;
  const numeric = sorted
    .map((c) => ({ c, n: parseChapterNumber(c.chapter) }))
    .filter((x): x is { c: Chapter; n: number } => x.n !== null);
  const atOrBelow = numeric.filter((x) => x.n <= lastRead + eps);
  if (atOrBelow.length > 0) return atOrBelow[atOrBelow.length - 1].c;
  return sorted[0];
}

export async function fetchChaptersForManga(
  mangaId: string,
): Promise<Chapter[]> {
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

export function hydrateMangaCovers(
  items: ReadingListItem[],
  setCoverUrls: Dispatch<SetStateAction<Record<string, string | null>>>,
): void {
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

export async function fetchMangaInfo(mangaId: string): Promise<MangaPublicInfo> {
  const empty: MangaPublicInfo = {
    title: null,
    description: null,
    availableTranslatedLanguages: [],
  };
  try {
    const res = await fetch(
      `${BACKEND_URL}/api/manga/${encodeURIComponent(mangaId)}/info`,
    );
    if (!res.ok) return empty;
    const j = (await res.json()) as Partial<MangaPublicInfo>;
    const langs = Array.isArray(j.availableTranslatedLanguages)
      ? j.availableTranslatedLanguages.filter(
          (x): x is string => typeof x === "string",
        )
      : [];
    return {
      title: typeof j.title === "string" ? j.title : null,
      description: typeof j.description === "string" ? j.description : null,
      availableTranslatedLanguages: langs,
    };
  } catch {
    return empty;
  }
}

export function coverFileNameFromUrl(coverUrl: string | null): string | undefined {
  if (!coverUrl) return undefined;
  const m = coverUrl.match(/\/covers\/[0-9a-f-]+\/([^/.]+)\.256\.jpg$/i);
  return m?.[1];
}

export function buildMangaForPopup(
  externalId: string,
  item: ReadingListItem,
  info: MangaPublicInfo | undefined,
  coverUrl: string | null,
): Manga {
  const titleStr = info?.title?.trim() || item.manga_title;
  const desc = info?.description?.trim() ?? "";
  const langs = info?.availableTranslatedLanguages ?? [];
  const fileName = coverFileNameFromUrl(coverUrl);
  const relationships: Manga["relationships"] = fileName
    ? [{ type: "cover_art", attributes: { fileName } }]
    : [];
  return {
    id: externalId,
    attributes: {
      title: { en: titleStr },
      description: desc ? { en: desc } : {},
      availableTranslatedLanguages: langs,
    },
    relationships,
  };
}

export async function hydrateMangaInfos(
  items: ReadingListItem[],
  setMangaInfos: Dispatch<
    SetStateAction<Record<string, MangaPublicInfo | undefined>>
  >,
): Promise<void> {
  const extIds = [
    ...new Set(items.map((i) => i.external_manga_id).filter(Boolean)),
  ] as string[];
  if (extIds.length === 0) {
    setMangaInfos({});
    return;
  }
  const entries = await Promise.all(
    extIds.map(async (id) => [id, await fetchMangaInfo(id)] as const),
  );
  setMangaInfos(Object.fromEntries(entries));
}
