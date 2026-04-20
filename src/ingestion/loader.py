"""
src/ingestion/loader.py
Load .txt, .pdf, and .docx documents from disk.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_document(filepath: str) -> str:
    """
    Load a single document from disk.
    Supports: .txt, .pdf, .docx
    Returns plain text string.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if not path.exists():
        logger.warning(f"File not found: {filepath}")
        return ""

    try:
        if suffix == ".txt":
            return _load_txt(path)
        elif suffix == ".pdf":
            return _load_pdf(path)
        elif suffix == ".docx":
            return _load_docx(path)
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return ""
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return ""


def _load_txt(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _load_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
        text = ""
        with fitz.open(str(path)) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
        return ""


def _load_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs)
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return ""


def load_all_documents(folder: str) -> dict:
    """
    Load all supported documents from a folder.
    Returns dict mapping filename -> text.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.warning(f"Folder not found: {folder}")
        return {}

    docs = {}
    for ext in ("*.txt", "*.pdf", "*.docx"):
        for filepath in folder_path.glob(ext):
            text = load_document(str(filepath))
            if text.strip():
                docs[filepath.name] = text
                logger.info(f"Loaded: {filepath.name} ({len(text)} chars)")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs
