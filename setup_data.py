"""
setup_data.py
Run this script once to generate all pre-processed data files
(equivalent to running all three notebooks).

Usage:
    python setup_data.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_PATH)

from src.ingestion.loader import load_all_documents
from src.ingestion.text_cleaner import clean_all_documents
from src.ingestion.document_segmentor import segment_document_into_clauses
from src.semantic.rule_classifier import classify_all_clauses
from src.risk.risk_scorer import score_all_clauses
from src.risk.explainer import apply_explanations
from src.semantic.embedder import generate_embeddings, save_embeddings
from src.risk.similarity_engine import find_similar_pairs, find_gap_clauses
from sklearn.feature_extraction.text import TfidfVectorizer

RAW_DATA_PATH  = Path("app/data/raw")
OUTPUT_PATH    = Path("app/outputs/risk_reports")
EMBEDDINGS_DIR = Path("app/embeddings")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("  Policy Compliance AI — Data Setup")
    print("=" * 60)

    # ── Step 1: Load & segment ─────────────────────────────────────
    print("\n[Step 1] Loading documents from app/data/raw/ …")
    docs = load_all_documents(str(RAW_DATA_PATH))
    if not docs:
        print("  ERROR: No documents found. Add .txt/.pdf/.docx files to app/data/raw/")
        sys.exit(1)

    cleaned = clean_all_documents(docs)
    all_clauses = []
    for doc_name, text in cleaned.items():
        clauses = segment_document_into_clauses(text, doc_name=doc_name)
        for c in clauses:
            c["document"] = doc_name
        all_clauses.extend(clauses)
        print(f"  {doc_name}: {len(clauses)} clauses")

    print(f"  Total: {len(all_clauses)} clauses")

    # ── Step 2: Classify + score + explain ─────────────────────────
    print("\n[Step 2] Classifying, scoring, and explaining …")
    all_clauses = classify_all_clauses(all_clauses)
    all_clauses = score_all_clauses(all_clauses)
    all_clauses = apply_explanations(all_clauses)

    df = pd.DataFrame(all_clauses)
    df["word_count"] = df["clause_text"].str.split().str.len()

    clauses_file = OUTPUT_PATH / "risk_labeled_clauses.csv"
    df.to_csv(clauses_file, index=False)
    print(f"  Saved: {clauses_file}")

    dist = df["risk_level"].value_counts().to_dict()
    print(f"  Risk distribution: {dist}")

    # ── Conflict & gap analysis via TF-IDF ─────────────────────────
    print("\n[Step 2b] Computing similarity for conflict detection …")
    texts = df["clause_text"].fillna("").tolist()
    vec   = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vec.fit_transform(texts).toarray()

    clause_list = df.to_dict("records")
    pairs = find_similar_pairs(clause_list, tfidf, threshold=0.55)
    conflict_df = pd.DataFrame(pairs)
    conflict_file = OUTPUT_PATH / "conflict_registry.csv"
    conflict_df.to_csv(conflict_file, index=False)
    print(f"  Saved: {conflict_file} ({len(conflict_df)} conflicts)")

    gaps    = find_gap_clauses(clause_list, tfidf, threshold=0.05)
    gap_df  = pd.DataFrame(gaps)
    gap_file = OUTPUT_PATH / "gap_analysis.csv"
    gap_df.to_csv(gap_file, index=False)
    print(f"  Saved: {gap_file} ({len(gap_df)} gaps)")

    # ── Step 3: Embeddings ─────────────────────────────────────────
    print("\n[Step 3] Generating sentence embeddings (this may take a minute) …")
    embeddings = generate_embeddings(texts, show_progress=True)
    ids        = df["clause_id"].tolist()

    emb_path = EMBEDDINGS_DIR / "clause_embeddings.npy"
    ids_path = EMBEDDINGS_DIR / "clause_ids.json"
    save_embeddings(embeddings, ids, str(emb_path), str(ids_path))
    print(f"  Saved embeddings: {emb_path} {embeddings.shape}")
    print(f"  Saved IDs:        {ids_path}")

    print("\n" + "=" * 60)
    print("  ✅ Setup complete!")
    print("  Launch the dashboard with:")
    print("     streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
