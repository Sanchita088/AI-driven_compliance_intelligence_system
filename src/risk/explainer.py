"""
src/risk/explainer.py
Generate human-readable risk explanations for scored clauses.
"""
import logging
from typing import List, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def generate_explanation(clause: Dict) -> str:
    """
    Return a 1–2 sentence plain-English explanation
    of why this clause received its risk level.
    """
    risk_level  = clause.get("risk_level", "LOW")
    clause_type = clause.get("clause_type", "general")
    has_penalty = clause.get("has_penalty", False)
    has_deadline = clause.get("has_deadline", False)
    has_law_ref  = clause.get("has_law_reference", False)
    score        = clause.get("risk_score", 0)

    parts = []

    if risk_level == "HIGH":
        parts.append("This clause carries HIGH risk")
        if clause_type == "prohibition":
            parts.append("because it contains an explicit prohibition that, if violated, may trigger enforcement action")
        elif clause_type == "obligation":
            parts.append("due to a mandatory obligation with potential legal consequences")
        if has_penalty:
            parts.append("; penalties or sanctions are referenced")
    elif risk_level == "MEDIUM":
        parts.append("This clause is flagged as MEDIUM risk")
        if clause_type == "obligation":
            parts.append("as it imposes a compliance obligation")
        elif clause_type == "condition":
            parts.append("because it sets a conditional requirement")
        if has_deadline:
            parts.append(" with a time-bound deadline")
    else:
        parts.append("This is a LOW risk clause of general advisory nature")
        if clause_type == "general":
            parts.append(" with no direct obligations, prohibitions, or penalties detected")

    if has_law_ref:
        parts.append(". A legal reference is cited, requiring accurate cross-referencing")

    parts.append(f" (score: {score}/100).")
    return "".join(parts)


def get_sample_row(df: pd.DataFrame) -> pd.Series:
    """Return a random HIGH-risk row, or any row if none exist."""
    high = df[df["risk_level"] == "HIGH"]
    return high.sample(1).iloc[0] if not high.empty else df.sample(1).iloc[0]


def apply_explanations(clauses: List[Dict]) -> List[Dict]:
    """Add 'explanation' field to every clause dict."""
    for clause in clauses:
        clause["explanation"] = generate_explanation(clause)
    logger.info(f"Generated explanations for {len(clauses)} clauses.")
    return clauses
