"""
src/ingestion/text_cleaner.py
Text normalization and cleaning utilities.
"""
import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Normalize raw document text:
    - Remove non-printable characters
    - Collapse excessive whitespace
    - Strip page markers inserted by PDF extractor
    - Normalize quotes and dashes
    """
    if not text:
        return ""

    # Remove page markers like [Page 1]
    text = re.sub(r'\[Page\s+\d+\]', '', text)

    # Normalize Unicode dashes and quotes to ASCII equivalents
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    # Remove non-printable / control characters (keep newlines)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\xA0-\xFF]', ' ', text)

    # Collapse 3+ blank lines into two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse spaces/tabs on a single line
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def clean_all_documents(docs: Dict[str, str]) -> Dict[str, str]:
    """Apply clean_text to every document in the dict."""
    cleaned = {}
    for name, text in docs.items():
        cleaned[name] = clean_text(text)
        logger.info(f"Cleaned: {name}")
    return cleaned
