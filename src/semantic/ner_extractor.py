"""
src/semantic/ner_extractor.py
Named Entity Recognition over clause texts using spaCy.
Falls back to regex-based extraction if spaCy is unavailable.
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

_nlp = None


def _load_spacy():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded: en_core_web_sm")
    except Exception as e:
        logger.warning(f"spaCy unavailable ({e}). Using regex fallback.")
        _nlp = None
    return _nlp


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text.
    Returns dict: {entity_type: [entity_texts]}.
    """
    nlp = _load_spacy()

    if nlp is not None:
        doc = nlp(text[:5000])  # cap length for speed
        entities: Dict[str, List[str]] = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return entities

    # ── Regex fallback ────────────────────────────────────────────────────────
    entities: Dict[str, List[str]] = {}

    # Dates / durations
    dates = re.findall(
        r'\b(\d{1,2}\s+\w+\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|within\s+\d+\s+\w+)\b',
        text, re.IGNORECASE
    )
    if dates:
        entities["DATE"] = list(set(dates))

    # Organisations (Title Case sequences of 2+ words)
    orgs = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4})\b', text)
    if orgs:
        entities["ORG"] = list(set(orgs[:10]))

    # Numeric references (Article N, Section N, etc.)
    refs = re.findall(
        r'(Article|Section|Regulation|Directive|Standard)\s+[\d\.]+',
        text, re.IGNORECASE
    )
    if refs:
        entities["LAW"] = list(set(refs))

    return entities


def run_ner_on_all_clauses(clauses: List[Dict]) -> List[Dict]:
    """Add 'entities' dict to each clause record."""
    for clause in clauses:
        text = clause.get("clause_text", clause.get("text", ""))
        clause["entities"] = extract_entities(text)
    logger.info(f"NER completed on {len(clauses)} clauses.")
    return clauses
