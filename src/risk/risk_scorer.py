"""
src/risk/risk_scorer.py
Multi-factor risk scoring for compliance clauses.
Score range: 0–100. Levels: HIGH ≥70, MEDIUM 40–69, LOW <40.
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── Keyword weights ───────────────────────────────────────────────────────────
_PROHIBITION_KW  = ['must not', 'shall not', 'prohibited', 'forbidden', 'not permitted',
                    'not allowed', 'no person shall', 'strictly prohibited']
_OBLIGATION_KW   = ['must', 'shall', 'required to', 'obliged', 'is obligatory',
                    'has a duty', 'is responsible for']
_PENALTY_KW      = ['fine', 'penalty', 'sanction', 'imprisonment', 'liable',
                    'prosecuted', 'offence', 'violation']
_DEADLINE_KW     = ['within', 'no later than', 'by the end of', 'before',
                    'deadline', 'not later than', 'prior to']
_LAW_REF_PATTERN = re.compile(
    r'(section|article|regulation|act|directive|standard|iso|iaea)\s*[\d\.]+',
    re.IGNORECASE
)


def _score_factors(text: str) -> dict:
    """Return individual risk factor scores for a clause."""
    tl = text.lower()
    factors = {
        "prohibition":  40 if any(k in tl for k in _PROHIBITION_KW) else 0,
        "penalty":      30 if any(k in tl for k in _PENALTY_KW)     else 0,
        "obligation":   20 if any(k in tl for k in _OBLIGATION_KW)  else 0,
        "deadline":     15 if any(k in tl for k in _DEADLINE_KW)    else 0,
        "law_ref":      10 if _LAW_REF_PATTERN.search(text)         else 0,
    }
    return factors


def score_clause(clause_text: str) -> Dict:
    """
    Score a single clause.
    Returns dict with risk_score (0–100), risk_level, clause_type, and flags.
    """
    if not clause_text:
        return {"risk_score": 0, "risk_level": "LOW", "clause_type": "general",
                "has_penalty": False, "has_deadline": False, "has_law_reference": False}

    tl = clause_text.lower()
    factors = _score_factors(clause_text)
    raw_score = sum(factors.values())
    risk_score = min(int(raw_score), 100)

    # Clause type
    if any(k in tl for k in _PROHIBITION_KW):
        clause_type = "prohibition"
    elif any(k in tl for k in _OBLIGATION_KW):
        clause_type = "obligation"
    elif any(k in tl for k in _DEADLINE_KW):
        clause_type = "condition"
    else:
        clause_type = "general"

    # Risk level
    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score":        risk_score,
        "risk_level":        risk_level,
        "clause_type":       clause_type,
        "has_penalty":       factors["penalty"] > 0,
        "has_deadline":      factors["deadline"] > 0,
        "has_law_reference": factors["law_ref"] > 0,
    }


def score_all_clauses(clauses: List[Dict]) -> List[Dict]:
    """
    Add risk fields to every clause dict in-place.
    Expects dicts with at least 'clause_text' key.
    """
    for clause in clauses:
        result = score_clause(clause.get("clause_text", ""))
        clause.update(result)
    logger.info(f"Scored {len(clauses)} clauses.")
    return clauses
