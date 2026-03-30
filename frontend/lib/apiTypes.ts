/**
 * JSON shapes shared with the FastAPI backend / MangaDex-style proxies.
 * Keep in sync when backend response contracts change.
 */
import type { Manga } from "@/app/types/types";

/** GET /api/manga/search and similar endpoints return a MangaDex-shaped payload. */
export type MangaSearchListJson = {
  data?: Manga[] | null;
};

/** GET /api/manga/chapter/{chapterId}/pages */
export type ChapterPagesJson = {
  urls?: string[];
};
