import httpx
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

client = httpx.AsyncClient(timeout=10.0)

# async def get_manga_page_stream(target_url: str):
#     try:
#         stream = client.stream("GET", target_url)
#         response = await stream.__aenter__()
#     except httpx.RequestError:
#         raise HTTPException(status_code=502, detail="Upstream request failed")

#     if response.status_code != 200:
#         await stream.__aexit__(None, None, None)
#         raise HTTPException(status_code=response.status_code, detail="Image not found")

#     async def generator():
#         try:
#             async for chunk in response.aiter_bytes():
#                 yield chunk
#         finally:
#             await stream.__aexit__(None, None, None)  # ensure cleanup

#     return StreamingResponse(
#         generator(),
#         media_type=response.headers.get("Content-Type", "image/jpeg"),
#     )

async def get_manga_page_stream(target_url: str):
    req = client.build_request("GET", target_url)

    try:
        response = await client.send(req, stream=True)
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Upstream unreachable")

    if response.status_code != 200:
        await response.aclose() # Must close if we aren't streaming
        raise HTTPException(status_code=response.status_code, detail="MangaDex error")

    return StreamingResponse(
        response.aiter_bytes(),
        media_type=response.headers.get("Content-Type", "image/jpeg"),
        background=response.aclose, # Ensures connection closes after stream finish
    )