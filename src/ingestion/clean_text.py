"""Clean and normalize extracted text from PDFs."""

import re


def clean_text(text: str) -> str:
    """
    Normalize whitespace and newlines.

    - Collapse multiple spaces/tabs to a single space.
    - Collapse 3+ newlines to 2.
    - Strip leading/trailing whitespace.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
