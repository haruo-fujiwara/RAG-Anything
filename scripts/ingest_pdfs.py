#!/usr/bin/env python3
"""
Ingest all PDFs (and PPTX via PDF conversion) in a directory into a single
RAG-Anything knowledge base. Supports GraphML (default) or Neo4j graph storage
via LightRAG env config.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import sys

from dotenv import load_dotenv

# Ensure local package imports take precedence over any installed version.
sys.path.append(str(Path(__file__).parent.parent))


def _configure_graph_backend(backend: str) -> None:
    if backend == "neo4j":
        os.environ["LIGHTRAG_GRAPH_STORAGE"] = "Neo4JStorage"


def _load_processed_file_paths(working_dir: str) -> set[str]:
    status_path = Path(working_dir) / "kv_store_doc_status.json"
    if not status_path.exists():
        return set()

    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: failed to read doc status store: {exc}")
        return set()

    processed = set()
    for value in data.values():
        if isinstance(value, dict) and value.get("file_path"):
            processed.add(str(value["file_path"]))
    return processed


def _should_skip(input_path: Path, per_doc_output: Path, processed: set[str]) -> bool:
    candidates = {str(input_path.resolve()), input_path.name}
    if input_path.suffix.lower() == ".pptx":
        expected_pdf = per_doc_output / f"{input_path.stem}.pdf"
        candidates.add(str(expected_pdf.resolve()))
        candidates.add(expected_pdf.name)
    return any(candidate in processed for candidate in candidates)


def _build_rag(api_key: str, base_url: str, working_dir: str, parser_name: str):
    # Delay heavy imports until after env configuration.
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    config = RAGAnythingConfig(
        working_dir=working_dir or "./rag_storage",
        parser=parser_name,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    if hasattr(config, "use_full_path"):
        config.use_full_path = True

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
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
        if image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
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
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts, model=embedding_model, api_key=api_key, base_url=base_url
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


async def _ingest_dir(rag, input_dir: Path, output_dir: Path) -> None:
    candidates = sorted(list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.pptx")))
    if not candidates:
        print(f"No PDFs or PPTX files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = _load_processed_file_paths(rag.config.working_dir)
    for input_path in candidates:
        per_doc_output = output_dir / input_path.stem
        per_doc_output.mkdir(parents=True, exist_ok=True)
        if _should_skip(input_path, per_doc_output, processed):
            print(f"Skipping already-processed file: {input_path.name}")
            continue
        file_path = input_path
        if input_path.suffix.lower() == ".pptx":
            from raganything.parser import Parser

            print(f"Converting {input_path.name} to PDF")
            file_path = Parser.convert_office_to_pdf(input_path, per_doc_output)
        print(f"Processing {file_path.name} -> {per_doc_output}")
        await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=str(per_doc_output),
            parse_method="auto",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest all PDFs in a directory into RAG-Anything."
    )
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument(
        "--working-dir", "-w", default="./rag_storage", help="RAG working directory"
    )
    parser.add_argument(
        "--output-dir", "-o", default="./output", help="Per-document output directory"
    )
    parser.add_argument(
        "--backend",
        choices=["graphml", "neo4j"],
        default="graphml",
        help="Graph storage backend (default: graphml)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Parser selection: mineru or docling",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "OpenAI API key is required (set LLM_BINDING_API_KEY or use --api-key)."
        )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    load_dotenv(dotenv_path=".env", override=False)
    _configure_graph_backend(args.backend)

    rag = _build_rag(args.api_key, args.base_url, args.working_dir, args.parser)

    asyncio.run(_ingest_dir(rag, input_dir, Path(args.output_dir)))


if __name__ == "__main__":
    main()
