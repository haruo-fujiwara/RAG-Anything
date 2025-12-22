import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
import os


async def load_existing_lightrag():
    # Set up API configuration
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "https://api.openai.com/v1"  # Optional

   # First, create or load existing LightRAG instance
    lightrag_working_dir = "./rag_storage"

    # Check if previous LightRAG instance exists
    if os.path.exists(lightrag_working_dir) and os.listdir(lightrag_working_dir):
        print("✅ Found existing LightRAG instance, loading...")
    else:
        print("❌ No existing LightRAG instance found, will create new one")

    # Create/load LightRAG instance with your configuration
    lightrag_instance = LightRAG(
        working_dir=lightrag_working_dir,
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
    )

    # Initialize storage (this will load existing data if available)
    await lightrag_instance.initialize_storages()
    await initialize_pipeline_status()

    # Define vision model function for image processing
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
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
        # Pure text format
        else:
            return lightrag_instance.llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Now use existing LightRAG instance to initialize RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,  # Pass existing LightRAG instance
        vision_model_func=vision_model_func,
        # Note: working_dir, llm_model_func, embedding_func, etc. are inherited from lightrag_instance
    )

    # Query existing knowledge base
    result = await rag.aquery(
        "ビデオリサーチ社の企業ロゴが見たいです。",
        #"マーケティングデータはどのような会社が提供していますか？", #"What data has been processed in this LightRAG instance?",
        mode="hybrid"
    )
    print("Query result:", result)

    # Add new multimodal document to existing LightRAG instance
    '''
    await rag.process_document_complete(
        file_path="path/to/new/multimodal_document.pdf",
        output_dir="./output"
    )
    '''

if __name__ == "__main__":
    asyncio.run(load_existing_lightrag())