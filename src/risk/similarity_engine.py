"""
src/risk/similarity_engine.py
Cross-document clause similarity and conflict detection.
"""
import logging
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

CONFLICT_THRESHOLD = 0.85   # above this → likely contradiction or duplicate
GAP_THRESHOLD      = 0.30   # below this → clause has no close match (potential gap)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return NxN cosine similarity matrix for a set of embeddings."""
    if embeddings is None or len(embeddings) == 0:
        return np.array([])
    return cosine_similarity(embeddings)


def find_similar_pairs(
    clauses: List[Dict],
    embeddings: np.ndarray,
    threshold: float = CONFLICT_THRESHOLD,
) -> List[Dict]:
    """
    Find clause pairs with similarity above threshold.
    Returns list of conflict records.
    """
    if len(embeddings) == 0 or len(clauses) != len(embeddings):
        return []

    sim_matrix = compute_similarity_matrix(embeddings)
    pairs = []
    n = len(clauses)

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim < threshold:
                continue

            ca, cb = clauses[i], clauses[j]
            # Skip pairs from the same document — focus on cross-doc conflicts
            if ca.get("document", ca.get("source", "")) == cb.get("document", cb.get("source", "")):
                continue

            rl_a = ca.get("risk_level", "LOW")
            rl_b = cb.get("risk_level", "LOW")

            # Determine conflict type
            type_a = ca.get("clause_type", "general")
            type_b = cb.get("clause_type", "general")

            if type_a == "prohibition" and type_b == "obligation":
                conflict_type = "Contradiction"
            elif type_a == "obligation" and type_b == "prohibition":
                conflict_type = "Contradiction"
            elif sim > 0.95:
                conflict_type = "Duplicate"
            else:
                conflict_type = "Overlap"

            severity = "HIGH" if conflict_type == "Contradiction" else \
                       "MEDIUM" if conflict_type == "Overlap" else "LOW"

            pairs.append({
                "clause_id_a":       ca.get("clause_id", f"C{i}"),
                "clause_id_b":       cb.get("clause_id", f"C{j}"),
                "document_a":        ca.get("document", ca.get("source", "")),
                "document_b":        cb.get("document", cb.get("source", "")),
                "text_a":            ca.get("clause_text", ca.get("text", ""))[:300],
                "text_b":            cb.get("clause_text", cb.get("text", ""))[:300],
                "similarity":        round(sim, 4),
                "conflict_type":     conflict_type,
                "conflict_severity": severity,
            })

    logger.info(f"Found {len(pairs)} similar/conflicting clause pairs.")
    return pairs


def find_gap_clauses(
    clauses: List[Dict],
    embeddings: np.ndarray,
    threshold: float = GAP_THRESHOLD,
) -> List[Dict]:
    """
    Identify obligation clauses that have no close semantic match
    elsewhere (potential compliance gaps).
    """
    if len(embeddings) == 0:
        return []

    sim_matrix = compute_similarity_matrix(embeddings)
    gaps = []

    for i, clause in enumerate(clauses):
        if clause.get("clause_type") not in ("obligation", "prohibition"):
            continue
        # Max similarity to any other clause
        row = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        if len(row) == 0:
            continue
        max_sim = float(row.max())
        if max_sim < threshold:
            gaps.append({
                "clause_id":       clause.get("clause_id", f"G{i}"),
                "document":        clause.get("document", clause.get("source", "")),
                "section":         clause.get("section", ""),
                "obligation_text": clause.get("clause_text", clause.get("text", ""))[:300],
                "gap_severity":    "HIGH" if clause.get("risk_level") == "HIGH" else "MEDIUM",
                "max_similarity":  round(max_sim, 4),
            })

    logger.info(f"Detected {len(gaps)} potential compliance gaps.")
    return gaps
