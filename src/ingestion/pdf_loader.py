"""Load PDF and extract text per page."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def load_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of {"page": int, "text": str}, 1-based page numbers.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i + 1, "text": text})
    return pages
