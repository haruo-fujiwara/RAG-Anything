import mimetypes

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from raganything.utils import compress_image_to_base64, extract_reference_paths

from mcp_stdio_server import init_rag, mcp, rag_ready

_mcp_http_app = mcp.http_app(path="/", transport="streamable-http")
app = FastAPI(title="RAG-Anything API", lifespan=_mcp_http_app.lifespan)
app.mount("/mcp", _mcp_http_app)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("hybrid", description="Query mode: hybrid, local, or global")
    image_response: str = Field(
        "paths", description="Image response format: paths or base64"
    )
    compressed: bool = Field(
        False, description="Compress images when image_response=base64"
    )
    max_images: int = Field(
        0, description="Max number of images to return (0 means no limit)"
    )
    max_image_kb: int = Field(900, description="Max image size in KB when compressed")
    max_image_px: int = Field(1024, description="Max image dimension in pixels")
    jpeg_quality: int = Field(70, description="Initial JPEG quality for compression")


@app.on_event("startup")
async def startup() -> None:
    await init_rag()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ready": rag_ready()}


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    try:
        rag = await init_rag()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = await rag.aquery(request.question, mode=request.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = {"answer": result}
    image_paths = getattr(rag, "_current_images_paths", [])
    if request.max_images > 0:
        image_paths = image_paths[: request.max_images]
    if image_paths:
        response["image_paths"] = image_paths
    reference_paths = extract_reference_paths(result)
    if reference_paths:
        response["reference_paths"] = reference_paths
    if request.image_response == "base64":
        images_base64 = getattr(rag, "_current_images_base64", [])
        if request.max_images > 0:
            images_base64 = images_base64[: request.max_images]
        content = [
            {
                "type": "text",
                "text": "Here are the relevant images found in the knowledge graph:",
            }
        ]
        for idx, image_data in enumerate(images_base64):
            mime_type = "image/jpeg"
            if idx < len(image_paths):
                guess, _ = mimetypes.guess_type(image_paths[idx])
                if guess:
                    mime_type = guess
            if request.compressed and idx < len(image_paths):
                try:
                    image_data = compress_image_to_base64(
                        image_paths[idx],
                        max_size_kb=request.max_image_kb,
                        max_dimension_px=request.max_image_px,
                        jpeg_quality=request.jpeg_quality,
                    )
                    mime_type = "image/jpeg"
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
            content.append(
                {
                    "type": "image",
                    "data": image_data,
                    "mimeType": mime_type,
                }
            )
        response["content"] = content
        response.pop("image_paths", None)
    return response
