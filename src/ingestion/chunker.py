"""Split text into overlapping chunks; optional semantic (paragraph-aware) chunking."""

import re
from typing import Any


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict[str, Any]]:
    """
    Sliding-window chunking by character count.

    Args:
        text: Input text to chunk.
        chunk_size: Target size per chunk in characters.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of {"text": str, "start": int, "end": int}.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        snippet = text[start:end]
        chunks.append({"text": snippet, "start": start, "end": end})
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraphs(
    text: str,
    chunk_size: int = 512,
    overlap_paragraphs: int = 1,
) -> list[dict[str, Any]]:
    """
    Chunk by grouping paragraphs so we rarely split mid-sentence.
    Splits on \\n\\n, then groups consecutive paragraphs until ~chunk_size.
    overlap_paragraphs: number of paragraphs to repeat at start of next chunk.

    Returns:
        List of {"text": str, "start": int, "end": int}.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return chunk_text(text, chunk_size=chunk_size, overlap=64)
    chunks = []
    i = 0
    while i < len(paragraphs):
        group = []
        length = 0
        j = i
        while j < len(paragraphs) and (length + len(paragraphs[j]) + 2 <= chunk_size or not group):
            group.append(paragraphs[j])
            length += len(paragraphs[j]) + 2
            j += 1
        if not group:
            group = [paragraphs[i]]
            j = i + 1
        chunk_text_str = "\n\n".join(group)
        start = text.find(chunk_text_str)
        if start < 0:
            start = sum(len(paragraphs[k]) + 2 for k in range(i))
        end = start + len(chunk_text_str)
        chunks.append({"text": chunk_text_str, "start": start, "end": end})
        i = j - overlap_paragraphs if overlap_paragraphs > 0 else j
        if i < j - 1:
            i = max(i, j - 1)
    return chunks
