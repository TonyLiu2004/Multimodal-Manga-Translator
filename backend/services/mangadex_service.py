import requests
import json
from cachetools import cached, TTLCache
import httpx

BASE_URL = "https://api.mangadex.org"


def _cover_url_from_manga_payload(data: dict, manga_id: str) -> str | None:
    """Build 256px cover URL from MangaDex /manga/{id} JSON (needs cover_art relationship)."""
    rels = data.get("relationships") or []
    for rel in rels:
        if rel.get("type") != "cover_art":
            continue
        attrs = rel.get("attributes") or {}
        fn = attrs.get("fileName")
        if isinstance(fn, str) and fn.strip():
            return f"https://uploads.mangadex.org/covers/{manga_id}/{fn}.256.jpg"
    return None


async def get_manga_cover_url_256(manga_id: str) -> str | None:
    """GET /manga/{id} with cover_art included; returns CDN URL or None."""
    url = f"{BASE_URL}/manga/{manga_id}"
    params = {"includes[]": ["cover_art"]}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError:
            return None
        except Exception as e:
            print(f"get_manga_cover_url_256: {e}")
            return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    return _cover_url_from_manga_payload(data, manga_id)


def _pick_localized_string(obj) -> str | None:
    if not isinstance(obj, dict) or not obj:
        return None
    en = obj.get("en")
    if isinstance(en, str) and en.strip():
        return en.strip()
    for v in obj.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _available_translated_languages(attrs: dict) -> list[str]:
    raw = attrs.get("availableTranslatedLanguages")
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if isinstance(x, str)]


async def get_manga_public_info(manga_id: str) -> dict:
    """Localized title, description, and chapter languages from GET /manga/{id}."""
    empty = {
        "title": None,
        "description": None,
        "availableTranslatedLanguages": [],
    }
    url = f"{BASE_URL}/manga/{manga_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            print(f"get_manga_public_info: {e}")
            return empty
    data = payload.get("data")
    if not isinstance(data, dict):
        return empty
    attrs = data.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    return {
        "title": _pick_localized_string(attrs.get("title")),
        "description": _pick_localized_string(attrs.get("description")),
        "availableTranslatedLanguages": _available_translated_languages(attrs),
    }


async def search_manga(
    title: str,
    limit: int = 20,
    offset: int = 0,
    order_by: str = "followedCount",
    order_direction: str = "desc",
    cover_art: bool = True,
    included_tags: list[str] | None = None,
    excluded_tags: list[str] | None = None,
):
    """Search MangaDex manga with optional tag filtering."""

    search_url = f"{BASE_URL}/manga"

    # Build params as list of tuples to support repeated keys (e.g. includedTags[])
    params: list[tuple[str, str]] = [
        ("limit", str(limit)),
        ("offset", str(offset)),
        (f"order[{order_by}]", order_direction),
    ]

    if cover_art:
        params.append(("includes[]", "cover_art"))

    if title.strip():
        params.append(("title", title))

    for tag_id in (included_tags or []):
        params.append(("includedTags[]", tag_id))

    for tag_id in (excluded_tags or []):
        params.append(("excludedTags[]", tag_id))

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(search_url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data
        except httpx.HTTPStatusError as e:
            return {"error": f"MangaDex API error: {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            print(f"Error fetching from MangaDex: {e}")
            return {"error": "Internal server error", "details": str(e)}
    return {"error": "end of function"}


async def get_manga_chapters(
    manga_id: str, 
    limit: int = 100,
    languages: list[str] = None,
    offset: int = 0,
    order_by: str = "chapter", 
    order_direction: str = "desc",
    content_rating: list[str] = None,
    include_empty: int = 0
):
    """
    Get the chapters of a given manga id
    """
    url = f"{BASE_URL}/manga/{manga_id}/feed"
    params = {
        "order[chapter]": "desc",
        "limit": limit,
        "offset": offset,
        "contentRating[]": content_rating,
        "includeEmptyPages": include_empty,
        f"order[{order_by}]": order_direction,
        "includes[]": ["scanlation_group"]
    }
    if languages:
        params["translatedLanguage[]"] = languages

    print(f"get_chapters called with params: {params}")

    url = f"https://api.mangadex.org/manga/{manga_id}/feed"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url, 
                params=params, 
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return  {"error": f"MangaDex API error: {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            return {"error": "Internal server error", "details": str(e)}


# Cache up to 100 panels, each for 300 seconds (5 minutes)
cache = TTLCache(maxsize=100, ttl=300)

@cached(cache)
def get_chapter_panel_urls(chapter_id: str, img_quality: str = "dataSaver"):
    """
    Fetches the actual image URLs for a given MangaDex Chapter ID.
    """
    print(f"[CACHE MISS] Fetching fresh URLs from MangaDex for: {chapter_id}")
    if img_quality not in ['data', 'dataSaver']:
        raise ValueError("img_quality must be 'data' or 'dataSaver'.")
    try:
        HEADERS = {
            "User-Agent": "Manglify (Capstone project) - https://github.com/TonyLiu2004/Multimodal-Manga-Translator"
        }
                
        r = requests.get(f"{BASE_URL}/at-home/server/{chapter_id}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    
        # 2. Grab the base URL and the chapter-specific hash
        base_url = data["baseUrl"]
        chapter_hash = data["chapter"]["hash"]
        file_names = data["chapter"][img_quality] # "data" is high quality, "dataSaver" is compressed

        # 3. Construct the full URL for every page
        # Format: {baseUrl}/data/{hash}/{filename}
        # - the "data" section can be "data" or "data-saver" depending on img_quality
        url_point = "data"
        if img_quality == "dataSaver":
            url_point = "data-saver"

        page_urls = [f"{base_url}/{url_point}/{chapter_hash}/{name}" for name in file_names]
        return page_urls
    except requests.exceptions.RequestException as err:
        print(f"HTTP error occurred: {err}")
    
    return None

### testing

# mangas = search_manga("", 15,  0, order={"followedCount": "desc"})
# print(mangas)
# first_manga_id = mangas[0]['id']
# chapters = get_chapters(first_manga_id)
# first_chapter_id = chapters[0]['id']
# first_chapter_panels = get_chapter_panel_urls(first_chapter_id, "dataSaver")
# print(first_manga_id)
# print(first_chapter_id)
# print(first_chapter_panels)
