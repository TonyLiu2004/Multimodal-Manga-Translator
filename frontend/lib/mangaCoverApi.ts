import { BACKEND_URL } from "@/config";

/** Resolve MangaDex catalog id to a 256px cover URL via backend (MangaDex API). */
export async function fetchMangaCoverUrl(
  mangaDexId: string,
): Promise<string | null> {
  const res = await fetch(
    `${BACKEND_URL}/api/manga/${encodeURIComponent(mangaDexId)}/cover`,
  );
  if (!res.ok) return null;
  const j = (await res.json()) as { cover_url?: string | null };
  return typeof j.cover_url === "string" && j.cover_url.length > 0
    ? j.cover_url
    : null;
}
