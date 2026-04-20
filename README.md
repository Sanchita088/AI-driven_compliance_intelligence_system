# AI-Driven Policy & Compliance Intelligence System

> **TCS iON Industry Project** | AKS University Satna | 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-drivencomplianceintelligencesystem-ew7r5jnbbenpprjjcttwyt.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-Sentence--Transformers-orange)


---

## 🔗 Live Demo

**👉 [Click here to open the app](https://ai-drivencomplianceintelligencesystem-ew7r5jnbbenpprjjcttwyt.streamlit.app/)**

---

## 📌 Overview

An end-to-end AI-powered compliance intelligence system that automatically processes policy and regulatory documents, extracts structured compliance knowledge, detects risks and conflicts, and delivers explainable insights through an interactive web interface.

Built for Governance, Risk & Compliance (GRC) teams to replace slow, error-prone manual policy review with intelligent automation.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Document Ingestion** | Upload PDF or TXT policy documents — automatic extraction and cleaning |
| 🔍 **Semantic Q&A** | Ask questions in plain English — answers grounded in source clauses |
| ⚠️ **Risk Scoring** | Every clause scored 0–100 with HIGH / MEDIUM / LOW label |
| 💡 **Explainability** | Plain-English reasoning for every risk score |
| ⚡ **Conflict Detection** | Cross-document clause contradiction and duplication detection |
| 🕳️ **Gap Analysis** | Obligations without corresponding control clauses flagged |
| 📝 **Document Summary** | AI-generated executive summary with obligations, entities, recommendations |
| ⬇️ **Export** | Download risk reports, clauses, and recommendations as CSV |

---

## 🖥️ Application Screens

```
🏠 Home Dashboard     → KPI metrics, risk charts, filters, drilldown table
📄 Document Explorer  → Clause-level browsing with risk metadata per document
💬 Compliance Q&A     → Semantic search over policy content
⚠️ Risk Dashboard     → Risk overview, conflict registry, gap analysis
📝 Document Summary   → AI summary with loader, risk profile, recommendations
```

---

## 🏗️ System Architecture

```
Raw PDFs
   │
   ▼ Step 1 — Ingestion
Clause Dataset (clause_id · text · section · metadata)
   │
   ▼ Step 2 — Semantic Understanding
Embeddings (384-dim) + NER Entities + Rule Classification
   │
   ▼ Step 3 — Risk Assessment
Risk Scores · Conflict Registry · Gap Analysis · Explanations
   │
   ▼ Step 4 — Interface
Streamlit App (5 screens · Q&A · Charts · CSV Export)
```

---

## 📁 Project Structure

```
AI-driven_compliance_intelligence_system/
│
├── 📁 data/
│   ├── raw/                        ← Original PDF documents
│   ├── extracted/                  ← Text extracted from PDFs
│   ├── cleaned/                    ← Normalized text
│   └── clauses/                    ← Segmented clause datasets
│
├── 📁 notebooks/
│   ├── step1_ingestion_preprocessing.ipynb
│   ├── step2_semantic_extraction.ipynb
│   ├── step3_risk_assessment.ipynb
│   └── step4_interface_demo.ipynb
│
├── 📁 src/
│   ├── ingestion/                  ← pdf_extractor · text_cleaner · document_segmentor
│   ├── semantic/                   ← embedder · ner_extractor · rule_classifier
│   ├── risk/                       ← similarity_engine · risk_scorer · explainer
│   └── interface/                  ← app.py (Streamlit)
│
├── 📁 embeddings/
│   ├── clause_embeddings.npy       ← 384-dim vectors for all clauses
│   └── clause_ids.json             ← Clause ID index
│
├── 📁 outputs/
│   └── risk_reports/
│       ├── risk_labeled_clauses.csv
│       ├── conflict_registry.csv
│       └── gap_analysis.csv
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-driven_compliance_intelligence_system.git
cd AI-driven_compliance_intelligence_system
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run the Streamlit app
```bash
streamlit run src/interface/app.py
```

App opens at **http://localhost:8501**

---

## 📓 Running the Pipeline (Notebooks)

Run notebooks **in order** before using pre-processed data in the app:

```
Step 1 → notebooks/step1_ingestion_preprocessing.ipynb
Step 2 → notebooks/step2_semantic_extraction.ipynb
Step 3 → notebooks/step3_risk_assessment.ipynb
Step 4 → notebooks/step4_interface_demo.ipynb
```

Each notebook produces output files consumed by the next step.

---

## 📦 Requirements

```
pdfplumber
PyPDF2
spacy
nltk
pandas
numpy
sentence-transformers
faiss-cpu
scikit-learn
streamlit
plotly
python-dotenv
jupyter
ipykernel
```

---

## 🔬 How It Works

### Step 1 — Document Ingestion
- PDFs extracted page by page using `pdfplumber`
- 8-stage text cleaning: encoding fix → whitespace → header removal → sentence boundaries
- Heading detection using regex patterns (numbered, ALL CAPS, Title Case, keywords)
- Each clause stored with: `clause_id · document · section · text · word_count`

### Step 2 — Semantic Understanding
- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions per clause)
- **NER**: spaCy extracts laws, organizations, deadlines, penalties
- **Rule Classification**: Pattern matching on modal verbs classifies each clause as:
  `obligation · prohibition · permission · condition · penalty · definition · recommendation`

### Step 3 — Risk Assessment
- **Similarity matrix**: Cosine similarity across all clause pairs
- **Conflict types**: `DIRECT CONFLICT · CROSS DOCUMENT · DUPLICATION · AMBIGUITY`
- **Risk scoring formula**:
  ```
  Score = clause_type_weight + entity_signals + conflict_weight + complexity
  Risk = HIGH (≥70) | MEDIUM (40–69) | LOW (<40)
  ```
- **Gap analysis**: Obligations without matching control clauses flagged
- **Explainability**: Structured plain-English reasoning per clause

### Step 4 — Interface
- Semantic Q&A: query → embed → cosine search → top-k results with citations
- Document Summary: 5-step animated loader + executive summary + recommendations
- All outputs downloadable as CSV

---

## 📊 Sample Results

| Metric | Value |
|---|---|
| Documents processed | 4 policy documents |
| Total clauses extracted | 247 |
| HIGH risk clauses | 38 (15.4%) |
| MEDIUM risk clauses | 82 (33.2%) |
| LOW risk clauses | 127 (51.4%) |
| Conflicts detected | 12 |
| Compliance gaps | 7 |
| Embedding dimensions | 384 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| NER | spaCy (`en_core_web_sm`) |
| Vector Search | Scikit-Learn |
| PDF Processing | pdfplumber, PyPDF2 |
| Data | Pandas, NumPy |
| Visualization | Plotly Express, Plotly Graph Objects |
| Interface | Streamlit |
| Development | Jupyter, VS Code, GitHub |
| Deployment | Streamlit Cloud |

---

## 🚀 Deployment

The app is deployed on **Streamlit Cloud**:

```
https://ai-drivencomplianceintelligencesystem-ew7r5jnbbenpprjjcttwyt.streamlit.app/
```

To deploy your own instance:
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set main file as `src/interface/app.py`
5. Click Deploy

---

## 📄 Project Info

| Field | Details |
|---|---|
| Project Title | AI-Driven Policy & Compliance Intelligence System |
| Company | TCS iON Digital Learning |
| Institute | AKS University Satna |
| Duration | 08 Apr 2026 – 27 May 2026 |
| Total Effort | 90 hours |

---

## 📚 References

- Reimers & Gurevych (2019). Sentence-BERT. EMNLP 2019.
- Johnson et al. (2019). Billion-scale similarity search with GPUs. IEEE.
- Honnibal & Montani (2017). spaCy 2. Software.
- GDPR — Regulation (EU) 2016/679
- ISO/IEC 27001:2022

---

## ⚖️ License

This project was developed as part of the TCS iON Industry Project program.  
© 2026 — For academic and educational use only.
