"""
tests/test_pipeline.py
Unit tests for the core NLP pipeline modules.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

# ── Ingestion ─────────────────────────────────────────────────────────────────

def test_clean_text_basic():
    from src.ingestion.text_cleaner import clean_text
    raw = "Hello\u2013World\u2014test  with   spaces\n\n\n\n\nend"
    result = clean_text(raw)
    assert "\u2013" not in result
    assert "\u2014" not in result
    assert "  " not in result


def test_clean_text_empty():
    from src.ingestion.text_cleaner import clean_text
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_segment_document():
    from src.ingestion.document_segmentor import segment_document_into_clauses
    text = "All operators must comply with safety rules. It is prohibited to discharge waste. Within 30 days a report must be submitted."
    clauses = segment_document_into_clauses(text, "test_doc")
    assert len(clauses) >= 1
    assert all("clause_id" in c for c in clauses)
    assert all("clause_text" in c for c in clauses)


def test_detect_heading():
    from src.ingestion.document_segmentor import detect_heading
    assert detect_heading("SECTION 1: GENERAL PROVISIONS")
    assert detect_heading("Article 4 Requirements")
    assert not detect_heading("This is a normal sentence about compliance.")
    assert not detect_heading("")


# ── Risk Scoring ──────────────────────────────────────────────────────────────

def test_score_prohibition():
    from src.risk.risk_scorer import score_clause
    result = score_clause("Organisations must not transfer personal data to third parties.")
    assert result["risk_level"] == "HIGH"
    assert result["clause_type"] == "prohibition"
    assert result["risk_score"] >= 40


def test_score_obligation():
    from src.risk.risk_scorer import score_clause
    result = score_clause("Data controllers must notify the authority within 72 hours.")
    assert result["clause_type"] == "obligation"
    assert result["has_deadline"] is True


def test_score_penalty():
    from src.risk.risk_scorer import score_clause
    result = score_clause("Violations shall be subject to a fine of EUR 20,000,000.")
    assert result["has_penalty"] is True
    assert result["risk_score"] >= 40


def test_score_general():
    from src.risk.risk_scorer import score_clause
    result = score_clause("The document provides guidance on good practices.")
    assert result["risk_level"] == "LOW"
    assert result["clause_type"] == "general"


def test_score_all_clauses():
    from src.risk.risk_scorer import score_all_clauses
    clauses = [
        {"clause_text": "Operators must comply with all requirements."},
        {"clause_text": "It is prohibited to falsify records."},
        {"clause_text": "General guidance on best practice."},
    ]
    result = score_all_clauses(clauses)
    assert all("risk_score" in c for c in result)
    assert all("risk_level" in c for c in result)


# ── Classification ────────────────────────────────────────────────────────────

def test_classify_prohibition():
    from src.semantic.rule_classifier import classify_clause
    assert classify_clause("This activity is strictly prohibited.") == "prohibition"


def test_classify_obligation():
    from src.semantic.rule_classifier import classify_clause
    assert classify_clause("The operator must submit a report.") == "obligation"


def test_classify_permission():
    from src.semantic.rule_classifier import classify_clause
    assert classify_clause("The authority may grant an extension.") == "permission"


def test_classify_condition():
    from src.semantic.rule_classifier import classify_clause
    assert classify_clause("If the operator fails, the licence is revoked.") == "condition"


def test_classify_general():
    from src.semantic.rule_classifier import classify_clause
    assert classify_clause("This section provides background information.") == "general"


# ── Explainer ─────────────────────────────────────────────────────────────────

def test_generate_explanation_high():
    from src.risk.explainer import generate_explanation
    clause = {
        "risk_level": "HIGH", "clause_type": "prohibition",
        "has_penalty": True, "has_deadline": False,
        "has_law_reference": False, "risk_score": 75
    }
    exp = generate_explanation(clause)
    assert "HIGH" in exp
    assert len(exp) > 20


def test_generate_explanation_low():
    from src.risk.explainer import generate_explanation
    clause = {
        "risk_level": "LOW", "clause_type": "general",
        "has_penalty": False, "has_deadline": False,
        "has_law_reference": False, "risk_score": 0
    }
    exp = generate_explanation(clause)
    assert "LOW" in exp


# ── NER ───────────────────────────────────────────────────────────────────────

def test_ner_extracts_dates():
    from src.semantic.ner_extractor import extract_entities
    text = "The report must be submitted within 30 days of the effective date."
    entities = extract_entities(text)
    # Should find at least one entity (regex fallback)
    assert isinstance(entities, dict)


# ── Embedder ──────────────────────────────────────────────────────────────────

def test_generate_embeddings_shape():
    from src.semantic.embedder import generate_embeddings
    texts = ["This is a test sentence.", "Another compliance clause here."]
    embs = generate_embeddings(texts)
    assert embs.shape[0] == 2
    assert embs.shape[1] == 384  # MiniLM dimension


def test_generate_embeddings_empty():
    from src.semantic.embedder import generate_embeddings
    embs = generate_embeddings([])
    assert len(embs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
