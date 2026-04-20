"""
src/semantic/rule_classifier.py
Rule-based clause type classification.
Types: obligation | prohibition | condition | permission | definition | general
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── Keyword rules (checked in priority order) ─────────────────────────────────
_RULES = [
    ("prohibition",  ['must not', 'shall not', 'may not', 'prohibited', 'forbidden',
                      'not permitted', 'not allowed', 'no person shall']),
    ("obligation",   ['must', 'shall', 'required to', 'is obliged', 'has a duty',
                      'is responsible for', 'it is mandatory']),
    ("permission",   ['may', 'is permitted', 'is allowed', 'is entitled', 'has the right']),
    ("condition",    ['if ', 'where ', 'unless ', 'provided that', 'in the event',
                      'subject to', 'on condition']),
    ("definition",   ['means ', 'refers to', 'is defined as', 'for the purposes of',
                      'in this context']),
]


def classify_clause(text: str) -> str:
    """
    Return the clause type label for a text string.
    """
    tl = text.lower()
    for label, keywords in _RULES:
        if any(kw in tl for kw in keywords):
            return label
    return "general"


def classify_all_clauses(clauses: List[Dict]) -> List[Dict]:
    """
    Add 'clause_type' to every clause dict based on keyword rules.
    Works in-place and returns the list.
    """
    for clause in clauses:
        text = clause.get("clause_text", clause.get("text", ""))
        clause["clause_type"] = classify_clause(text)
    logger.info(f"Classified {len(clauses)} clauses.")
    return clauses
