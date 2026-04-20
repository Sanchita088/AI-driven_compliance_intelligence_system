"""
src/ingestion/pdf_extractor.py
Dedicated PDF text extraction utilities.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.
    Falls back to empty string on failure.
    """
    try:
        import fitz
        text_parts = []
        with fitz.open(filepath) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
        return "\n\n".join(text_parts)
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction error ({filepath}): {e}")
        return ""


def extract_all_pdfs(folder: str) -> Dict[str, str]:
    """
    Extract text from every PDF in a folder.
    Returns dict mapping filename -> extracted text.
    """
    folder_path = Path(folder)
    results = {}

    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder}")
        return results

    for pdf_file in sorted(folder_path.glob("*.pdf")):
        text = extract_text_from_pdf(str(pdf_file))
        if text.strip():
            results[pdf_file.name] = text
            logger.info(f"Extracted PDF: {pdf_file.name} ({len(text)} chars)")
        else:
            logger.warning(f"Empty extraction: {pdf_file.name}")

    return results


def get_pdf_metadata(filepath: str) -> dict:
    """Return basic metadata dict for a PDF file."""
    try:
        import fitz
        with fitz.open(filepath) as doc:
            meta = doc.metadata or {}
            return {
                "title":    meta.get("title", ""),
                "author":   meta.get("author", ""),
                "pages":    doc.page_count,
                "filepath": filepath,
            }
    except Exception:
        return {"filepath": filepath, "pages": 0, "title": "", "author": ""}
