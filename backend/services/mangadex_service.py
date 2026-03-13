import requests
import json
from cachetools import cached, TTLCache

BASE_URL = "https://api.mangadex.org"

def search_manga(title: str, limit: int =20, offset: int = 0, order: dict = {"relevance": "desc"}):
    """
    Todo: filters by tags (include, exclude)
    Mostly for testing at the moment
    """

    search_url = f"{BASE_URL}/manga"

    final_order_query = {}
    for key, value in order.items():
        final_order_query[f"order[{key}]"] = value

    search_params = {
        "title": title,
        "limit": limit,
        "offset": offset,
        "includes[]": ["cover_art"],
        **final_order_query
    }

    try:
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched {len(data.get("data", []))} results from MangaDex")
    except Exception as e:
        print(f"Error fetching from MangaDex: {e}")
        return []
    
    mangas = []
    for manga in data.get("data", []):
        id = manga['id']
        title = manga["attributes"]['title'].get("en", "")
        cover_art_url = None
        if title == "":
            title_values = list(manga["attributes"]["title"].values())
            title = title_values[0]

        cover_art_json = None
        for rel in manga.get("relationships", []):
            if rel["type"] == "cover_art":
                cover_art_json = rel
                break
        
        if cover_art_json:
            cover_art_filename = cover_art_json['attributes']['fileName']
            cover_art_url = f"https://uploads.mangadex.org/covers/{id}/{cover_art_filename}.256.jpg"
        
        # print(id)
        # print(title)
        # print(cover_art_url)
        mangas.append({
            "id": id,
            "title": title,
            "cover_url": cover_art_url
        })
    return mangas

def get_chapters(manga_id: str, limit: int = 5):
    """
    Get the chapters of a given manga id
    """
    feed_url = f"{BASE_URL}/manga/{manga_id}/feed"
    feed_params = {
        "translatedLanguage[]": ["en"],
        "order[chapter]": "desc",
        "limit": limit,
        "contentRating[]": ["safe", "suggestive"], #, "erotica", "pornographic"], #oh hell naw
        "includeEmptyPages": 0 #if pages = 0 or theres an externalURL, it means the manga is not hosted on mangedex so we can't download the images.
    }

    try:
        response = requests.get(feed_url, params=feed_params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            return []
        #if data is [], then the manga panels are probably hosted somewhere else and must be accessed through externalUrl. To see, put includeEmptyPages = 1
        return data["data"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Mangadex chapters for manga_id {manga_id}")
        return []


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

# mangas = search_manga("b", 1)
# print(mangas)
# first_manga_id = mangas[0]['id']
# chapters = get_chapters(first_manga_id)
# first_chapter_id = chapters[0]['id']
# first_chapter_panels = get_chapter_panel_urls(first_chapter_id, "dataSaver")
# print(first_manga_id)
# print(first_chapter_id)
# print(first_chapter_panels)
