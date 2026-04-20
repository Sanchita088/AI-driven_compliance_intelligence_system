# 🛡️ Policy Compliance AI

> **AI-Driven Policy & Compliance Intelligence System** — Automatic compliance extraction, risk detection, semantic Q&A, and document summarization for regulatory documents.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Policy Compliance AI is an end-to-end NLP pipeline that ingests regulatory documents (PDF, DOCX, TXT), extracts compliance clauses, scores risk levels, detects conflicts and gaps, and provides a semantic Q&A interface — all surfaced through an interactive Streamlit dashboard.

**Datasets Used:**
- 📄 Dataset 1: Development of Regulatory Documents
- 📄 Dataset 2: IAEA Safety Standards Series

---

## Features

| Feature | Description |
|---|---|
| 📥 Document Ingestion | PDF, DOCX, TXT support with automated text extraction |
| 🔍 Clause Extraction | Rule-based + ML extraction of obligations, prohibitions, deadlines |
| ⚠️ Risk Scoring | Multi-factor risk scoring (0–100) with HIGH / MEDIUM / LOW levels |
| 🧠 Semantic Q&A | Sentence-BERT embeddings + cosine similarity search |
| 🔴 Conflict Detection | Cross-document contradiction identification |
| 🕳️ Gap Analysis | Missing provision detection against compliance templates |
| 📊 Dashboard | Interactive Plotly charts, drilldown, and CSV export |
| 📝 AI Summary | Auto-generated executive summaries with recommendations |

---

## Project Structure

```
policy-compliance-ai/
│
├── app/                          # Streamlit application
│   ├── app.py                    # Main dashboard (5 tabs)
│   ├── embeddings/               # Saved sentence embeddings (.npy)
│   ├── outputs/
│   │   └── risk_reports/         # CSV outputs from notebooks
│   │       ├── risk_labeled_clauses.csv
│   │       ├── conflict_registry.csv
│   │       └── gap_analysis.csv
│   └── data/
│       └── raw/                  # Sample policy documents
│
├── src/                          # Core NLP modules
│   ├── ingestion/
│   │   ├── loader.py             # Document loading (PDF/DOCX/TXT)
│   │   ├── pdf_extractor.py      # PDF text extraction
│   │   ├── text_cleaner.py       # Text normalization
│   │   └── document_segmentor.py # Clause segmentation
│   ├── risk/
│   │   ├── risk_scorer.py        # Risk scoring logic
│   │   ├── explainer.py          # Risk explanation generator
│   │   └── similarity_engine.py  # Cross-doc similarity
│   └── semantic/
│       ├── embedder.py           # Sentence-BERT embeddings
│       ├── ner_extractor.py      # Named entity recognition
│       └── rule_classifier.py    # Rule-based clause classifier
│
├── notebooks/
│   ├── step1_ingestion.ipynb     # Document loading & cleaning
│   ├── step2_risk_scoring.ipynb  # Risk labeling pipeline
│   └── step3_embeddings.ipynb    # Embedding generation
│
├── tests/
│   └── test_pipeline.py          # Unit tests
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/policy-compliance-ai.git
cd policy-compliance-ai
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# OR using conda
conda create -n compliance-ai python=3.10
conda activate compliance-ai
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLP Models (first run)

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -m spacy download en_core_web_sm
```

---

## Quick Start

### Option A — Run with Sample Data (Recommended)

```bash
# 1. Run all preprocessing notebooks
jupyter nbconvert --to notebook --execute notebooks/step1_ingestion.ipynb
jupyter nbconvert --to notebook --execute notebooks/step2_risk_scoring.ipynb
jupyter nbconvert --to notebook --execute notebooks/step3_embeddings.ipynb

# 2. Launch the dashboard
streamlit run app/app.py
```

Open your browser at **http://localhost:8501**

### Option B — Upload Your Own Documents

```bash
streamlit run app/app.py
```

Then use the **sidebar → Upload your documents** to upload PDF / DOCX / TXT files directly.

---

## Usage Guide

### Tab 1 — Home Dashboard
- KPI cards: total clauses, high-risk count, conflicts, gaps
- Filter by document, risk level, and clause type
- Interactive pie chart, bar chart, gauge, and stacked breakdown

### Tab 2 — Document Explorer
- Select any loaded document
- View clause-level detail with risk badges
- Filter by clause type and risk level

### Tab 3 — Compliance Q&A
- Ask plain-English questions (e.g., *"What must we do after a data breach?"*)
- Semantic search returns top-K most relevant clauses with similarity scores
- Sample questions provided for quick exploration

### Tab 4 — Risk Dashboard
- Score distribution histogram
- Conflict registry with clause-pair view
- Gap analysis with severity badges
- Download high-risk clauses as CSV

### Tab 5 — Document Summary
- AI-generated executive summary
- Risk profile sub-tab with pie chart
- Clause breakdown with filtering
- Recommendations with action items + CSV export

---

## Notebooks

Run them **in order** before launching the app:

| Notebook | Purpose | Output |
|---|---|---|
| `step1_ingestion.ipynb` | Load & clean documents | Cleaned text files |
| `step2_risk_scoring.ipynb` | Extract, score, label clauses | `risk_labeled_clauses.csv` |
| `step3_embeddings.ipynb` | Generate sentence embeddings | `clause_embeddings.npy`, `clause_ids.json` |

---

## Dataset

Place your raw documents in `app/data/raw/`. Supported formats:

- `.txt` — Plain text regulatory documents
- `.pdf` — Scanned or digital PDFs (uses PyMuPDF)
- `.docx` — Microsoft Word documents

Sample documents from IAEA Safety Standards are included for demo purposes.

---

## Environment Variables

Copy `.env.example` to `.env` and set any optional keys:

```bash
cp .env.example .env
```

```env
# Optional: HuggingFace cache location
HF_HOME=./models/cache

# Optional: Logging level
LOG_LEVEL=INFO
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` embedding model
- [Streamlit](https://streamlit.io/) — Dashboard framework
- [Plotly](https://plotly.com/python/) — Interactive charts
- [spaCy](https://spacy.io/) — NER pipeline
- IAEA Safety Standards Series — Sample dataset
