import requests
import json

BASE_URL = "https://api.mangadex.org"

def search_manga(title, limit=20, offset=0, order={"relevance": "desc"}):
    """
    Todo: filters by tags (include, exclude)
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
        
        print(id)
        print(title)
        print(cover_art_url)
    
search_manga("One piece", 1)