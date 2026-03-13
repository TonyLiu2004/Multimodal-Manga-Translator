import httpx
from fastapi.responses import StreamingResponse

client = httpx.AsyncClient()

async def get_manga_page_stream(target_url: str):
    response = await client.get(target_url)
    return StreamingResponse(
        response.iter_bytes(), 
        media_type=response.headers.get("Content-Type", "image/jpeg")
    )