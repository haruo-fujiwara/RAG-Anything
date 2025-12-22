import asyncio
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from mcp_server import mcp

app = FastAPI(title="RAG-Anything API")
mcp_http_app = mcp.http_app(path="/", transport="streamable-http")
app.mount("/mcp", mcp_http_app)

_rag_instance: Optional[RAGAnything] = None
_rag_lock = asyncio.Lock()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("hybrid", description="Query mode: hybrid, local, or global")


async def _init_rag() -> RAGAnything:
    global _rag_instance
    async with _rag_lock:
        if _rag_instance is not None:
            return _rag_instance

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_dim = int(os.getenv("OPENAI_EMBEDDING_DIM", "3072"))
        working_dir = os.getenv("RAG_WORKING_DIR", "./rag_storage")

        lightrag_instance = LightRAG(
            working_dir=working_dir,
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            ),
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=embedding_model,
                    api_key=api_key,
                    base_url=base_url,
                ),
            ),
        )

        await lightrag_instance.initialize_storages()
        await initialize_pipeline_status()

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            if image_data:
                return openai_complete_if_cache(
                    vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            return lightrag_instance.llm_model_func(
                prompt, system_prompt, history_messages, **kwargs
            )

        _rag_instance = RAGAnything(
            lightrag=lightrag_instance,
            vision_model_func=vision_model_func,
        )
        return _rag_instance


@app.on_event("startup")
async def startup() -> None:
    await _init_rag()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ready": _rag_instance is not None}


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    try:
        rag = await _init_rag()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = await rag.aquery(request.question, mode=request.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": result}
