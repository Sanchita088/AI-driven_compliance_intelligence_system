import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Patterns that indicate section headings
_HEADING_PATTERNS = [
    re.compile(r'^\s*(\d+[\.\d]*)\s+[A-Z][A-Za-z\s]+$'),   # "1.2 Title"
    re.compile(r'^\s*[A-Z][A-Z\s]{4,}$'),                    # "SECTION TITLE"
    re.compile(r'^\s*(Article|Section|Chapter|Part)\s+\d+',   # Article / Section N
               re.IGNORECASE),
]


def detect_heading(line: str) -> bool:
    """Return True if the line looks like a section heading."""
    line = line.strip()
    if not line or len(line) > 120:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def segment_document_into_clauses(text: str, doc_name: str = "doc") -> List[Dict]:
    clauses = []
    current_section = "Preamble"
    clause_idx = 0

    # Split into lines, then sentences
    lines = text.split('\n')
    sentence_buffer = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if detect_heading(line_stripped):
            current_section = line_stripped
            continue
        sentence_buffer.append(line_stripped)

    full_text = " ".join(sentence_buffer)

    # Split into sentences on . ! ?
    raw_sentences = re.split(r'(?<=[.!?])\s+', full_text)

    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < 15:           # skip noise
            continue
        clauses.append({
            "clause_id":   f"{doc_name}_{clause_idx:04d}",
            "section":     current_section,
            "clause_text": sent,
            "source":      doc_name,
        })
        clause_idx += 1

    logger.info(f"Segmented '{doc_name}' → {len(clauses)} clauses")
    return clauses
