"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

import base64
import io
import os
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path
from lightrag.utils import logger


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_items.append(item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        resolved_path = resolve_media_path(image_path)
        with open(resolved_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = resolve_media_path(image_path)

        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


def resolve_media_path(media_path: str) -> Path:
    """
    Resolve a media path using MEDIA_BASE_PATH or OUTPUT_DIR when given a relative path.
    """
    path = Path(media_path)
    if path.is_absolute():
        return path
    base = os.getenv("MEDIA_BASE_PATH") or os.getenv("OUTPUT_DIR") or "./output"
    return (Path(base) / path).resolve()


def extract_reference_paths(answer: str) -> List[str]:
    """
    Extract reference file paths from an answer string if a References/Sources block exists.

    Args:
        answer: Full answer text (may include reference block).

    Returns:
        List of reference paths in the order they appear.
    """
    if not answer:
        return []

    pattern = re.compile(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(References?|Sources?)\s*:?",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(answer))
    if not matches:
        return []

    start = matches[-1].end()
    refs_block = answer[start:].strip()
    if not refs_block:
        return []

    # Handle bracketed list: [a, b]
    if refs_block.startswith("[") and refs_block.endswith("]"):
        inner = refs_block[1:-1].strip()
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]

    refs = []
    for line in refs_block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•\d\)\.\s]+", "", line).strip()
        line = re.sub(r"^\[\d+\]\s*", "", line)
        if line:
            refs.append(line)
    return refs


def compress_image_to_base64(
    image_path: str,
    max_size_kb: int = 900,
    max_dimension_px: int = 1024,
    jpeg_quality: int = 70,
) -> str:
    """
    Compress an image to JPEG and return base64 with size constraints.

    Args:
        image_path: Path to the image (relative or absolute).
        max_size_kb: Target max size in KB.
        max_dimension_px: Max width/height in pixels.
        jpeg_quality: Initial JPEG quality (1-95).

    Returns:
        Base64 encoded JPEG string.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image compression.") from exc

    resolved_path = resolve_media_path(image_path)
    with Image.open(resolved_path) as img:
        img = img.convert("RGB")
        if max_dimension_px and max(img.size) > max_dimension_px:
            img.thumbnail((max_dimension_px, max_dimension_px), Image.LANCZOS)

        quality = max(1, min(95, int(jpeg_quality)))
        for _ in range(10):
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= max_size_kb * 1024 or quality <= 30:
                return base64.b64encode(data).decode("utf-8")
            quality -= 10

        return base64.b64encode(data).decode("utf-8")


async def insert_text_content(
    lightrag,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("Text content insertion complete")


async def insert_text_content_with_multimodal_content(
    lightrag,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    try:
        await lightrag.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])
