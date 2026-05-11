import { BACKEND_URL } from "@/config";

/** Resolve MangaDex catalog id to a 256px cover URL via backend (MangaDex API). */
export async function fetchMangaCoverUrl(
  mangaDexId: string,
): Promise<string | null> {
  return `${BACKEND_URL}/api/manga/${encodeURIComponent(mangaDexId)}/cover-image`;
}
