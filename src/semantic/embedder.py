"""
src/semantic/embedder.py
Sentence-BERT embedding generation and nearest-neighbour retrieval.
"""
import os
import json
import logging
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_model_cache: Dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def generate_embeddings(
    texts: List[str],
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of strings into a 2-D numpy array of shape (N, D).
    """
    if not texts:
        return np.array([])
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    logger.info(f"Generated embeddings: {embeddings.shape}")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    clause_ids: List[str],
    emb_path: str,
    ids_path: str,
) -> None:
    """Persist embeddings (.npy) and clause IDs (JSON) to disk."""
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    np.save(emb_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(clause_ids, f)
    logger.info(f"Saved embeddings → {emb_path}")


def load_embeddings(emb_path: str, ids_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load previously saved embeddings and IDs from disk."""
    embeddings = np.load(emb_path) if os.path.exists(emb_path) else np.array([])
    clause_ids = json.load(open(ids_path)) if os.path.exists(ids_path) else []
    return embeddings, clause_ids


def find_similar_clauses(
    query: str,
    embeddings: np.ndarray,
    clauses: List[Dict],
    top_k: int = 5,
    model_name: str = _DEFAULT_MODEL,
) -> List[Dict]:
    """
    Semantic search: return top-k clause dicts most similar to query.
    """
    if embeddings is None or len(embeddings) == 0:
        return []

    model = _get_model(model_name)
    q_emb = model.encode([query], convert_to_numpy=True)
    sims  = cosine_similarity(q_emb, embeddings)[0]
    top_i = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_i:
        record = dict(clauses[idx])
        record["similarity"] = round(float(sims[idx]), 4)
        results.append(record)
    return results
