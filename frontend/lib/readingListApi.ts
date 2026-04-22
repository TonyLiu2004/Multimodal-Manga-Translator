import { BACKEND_URL } from "@/config";

export type ReadingListCollection = {
  id: number;
  name: string;
  manga_count: number;
  created_at: string | null;
  updated_at: string | null;
  /** MangaDex id of the most recently added title (for list cover). */
  latest_external_manga_id?: string | null;
};

export type ReadingListItem = {
  id: number;
  reading_list_id: number;
  manga_id: number;
  manga_title: string;
  external_manga_id: string | null;
  last_chapter_number: number | null;
  updated_at: string | null;
};

async function readError(res: Response): Promise<string> {
  const t = await res.text();
  if (!t) return res.statusText;
  try {
    const j = JSON.parse(t) as { detail?: string | unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail) && j.detail.length > 0) {
      const first = j.detail[0];
      if (typeof first === "object" && first && "msg" in first) {
        return String((first as { msg: string }).msg);
      }
      try {
        return JSON.stringify(j.detail);
      } catch {
        /* ignore */
      }
    }
  } catch {
    /* ignore */
  }
  return t;
}

function authHeaders(accessToken: string): HeadersInit {
  return { Authorization: `Bearer ${accessToken}` };
}

export async function fetchReadingLists(
  accessToken: string,
): Promise<ReadingListCollection[]> {
  const res = await fetch(`${BACKEND_URL}/reading-lists`, {
    headers: authHeaders(accessToken),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<ReadingListCollection[]>;
}

export async function createReadingList(
  accessToken: string,
  name: string,
): Promise<ReadingListCollection> {
  const res = await fetch(`${BACKEND_URL}/reading-lists`, {
    method: "POST",
    headers: {
      ...authHeaders(accessToken),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<ReadingListCollection>;
}

export async function renameReadingList(
  accessToken: string,
  readingListId: number,
  name: string,
): Promise<ReadingListCollection> {
  const res = await fetch(`${BACKEND_URL}/reading-lists/${readingListId}`, {
    method: "PATCH",
    headers: {
      ...authHeaders(accessToken),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<ReadingListCollection>;
}

export async function deleteReadingList(
  accessToken: string,
  readingListId: number,
): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/reading-lists/${readingListId}`, {
    method: "DELETE",
    headers: authHeaders(accessToken),
  });
  if (!res.ok) throw new Error(await readError(res));
}

export async function fetchReadingListItems(
  accessToken: string,
  readingListId: number,
): Promise<ReadingListItem[]> {
  const res = await fetch(
    `${BACKEND_URL}/reading-lists/${readingListId}/items`,
    { headers: authHeaders(accessToken) },
  );
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<ReadingListItem[]>;
}

export async function removeReadingListItem(
  accessToken: string,
  readingListId: number,
  mangaId: number,
): Promise<void> {
  const res = await fetch(
    `${BACKEND_URL}/reading-lists/${readingListId}/items/${mangaId}`,
    { method: "DELETE", headers: authHeaders(accessToken) },
  );
  if (!res.ok) throw new Error(await readError(res));
}

export async function addToReadingList(
  accessToken: string,
  params: {
    readingListId: number;
    external_manga_id: string;
    manga_title: string;
  },
): Promise<ReadingListItem> {
  const res = await fetch(
    `${BACKEND_URL}/reading-lists/${params.readingListId}/items`,
    {
      method: "POST",
      headers: {
        ...authHeaders(accessToken),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        provider_id: "mangadex",
        external_manga_id: params.external_manga_id,
        manga_title: params.manga_title,
      }),
    },
  );
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<ReadingListItem>;
}

/** Sync display name to backend `public.users` (after Supabase Auth user_metadata is updated). */
export async function patchAppUserDisplayName(
  accessToken: string,
  displayName: string,
): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/users/me`, {
    method: "PATCH",
    headers: {
      ...authHeaders(accessToken),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ display_name: displayName }),
  });
  if (!res.ok) throw new Error(await readError(res));
}
