import requests
import json
from cachetools import cached, TTLCache
import httpx

BASE_URL = "https://api.mangadex.org"

async def search_manga(title: str, limit: int =20, offset: int = 0, order_by: str = "followedCount", order_direction: str = "desc", cover_art: bool = True):
    """
    Todo: filters by tags (include, exclude)
    Mostly for testing at the moment
    """

    search_url = f"{BASE_URL}/manga"

    params = {
        "limit": limit,
        "offset": offset,
        f"order[{order_by}]": order_direction,
    }

    if cover_art:
        params["includes[]"] = ["cover_art"]

    if title.strip():
            params["title"] = title

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
