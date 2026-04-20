import os
import sys
import time
import tempfile
import re
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_PATH)

from src.ingestion.loader import load_document, load_all_documents
from src.ingestion.text_cleaner import clean_text
from src.risk.explainer import generate_explanation, apply_explanations
from src.risk.risk_scorer import score_clause, score_all_clauses
from src.risk.similarity_engine import compute_similarity_matrix, find_similar_pairs
from src.semantic.embedder import generate_embeddings, find_similar_clauses
from src.semantic.ner_extractor import extract_entities, run_ner_on_all_clauses
from src.semantic.rule_classifier import classify_clause, classify_all_clauses

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Policy Compliance AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── File paths ────────────────────────────────────────────────────────────────
APP_DIR          = Path(__file__).parent
STORE_PATH       = APP_DIR / "embeddings" / "clause_embeddings"
SAMPLE_DOCS_PATH = APP_DIR / "data" / "raw"
CLAUSES_PATH     = APP_DIR / "outputs" / "risk_reports" / "risk_labeled_clauses.csv"
EMBEDDINGS_PATH  = APP_DIR / "embeddings" / "clause_embeddings.npy"
IDS_PATH         = APP_DIR / "embeddings" / "clause_ids.json"
CONFLICT_PATH    = APP_DIR / "outputs" / "risk_reports" / "conflict_registry.csv"
GAP_PATH         = APP_DIR / "outputs" / "risk_reports" / "gap_analysis.csv"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1f3c88 0%, #2d6a9f 100%);
        color: white; padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ffeaea; border-left: 5px solid #e63946;
        padding: 10px 14px; border-radius: 5px; margin: 5px 0; color: #7b1a1a;
    }
    .risk-medium {
        background-color: #fff8e6; border-left: 5px solid #f4a261;
        padding: 10px 14px; border-radius: 5px; margin: 5px 0; color: #7d5200;
    }
    .risk-low {
        background-color: #eafaf1; border-left: 5px solid #2a9d8f;
        padding: 10px 14px; border-radius: 5px; margin: 5px 0; color: #1a5e35;
    }
    .summary-header {
        background: #f0f4ff; border-left: 4px solid #2d6a9f;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
        color: #1a2a4a; line-height: 1.6;
    }
    .kpi-card {
        background: var(--secondary-background-color);
        border-radius: 10px; padding: 14px 18px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "documents":     {},
    "all_chunks":    [],
    "all_clauses":   [],
    "faiss_index":   None,
    "index_metadata": [],
    "qa_history":    [],
    "chat_history":  [],
    "processed":     False,
    "embeddings_map": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

RISK_COLORS = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#639922"}


# HELPER FUNCTIONS


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 40) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_compliance_clauses(text: str) -> list:
    clauses = []
    sentences = re.split(r"(?<=[.!?]) +", text)
    prohibition_kw = ["must not", "shall not", "prohibited", "forbidden", "not allowed"]
    obligation_kw  = ["must", "shall", "required", "mandatory", "obliged"]
    deadline_kw    = ["within", "by", "before", "no later than", "deadline"]

    for i, s in enumerate(sentences):
        sl = s.lower()
        if any(k in sl for k in prohibition_kw):
            ctype, risk = "prohibition", "High"
        elif any(k in sl for k in obligation_kw):
            ctype, risk = "obligation", "Medium"
        elif any(k in sl for k in deadline_kw):
            ctype, risk = "deadline", "Medium"
        else:
            ctype, risk = "general", "Low"
        clauses.append({"clause_id": f"C{i:04d}", "text": s, "risk": risk, "type": ctype})
    return clauses


def extract_keywords(text: str, top_n: int = 10) -> list:
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=top_n)
        vec.fit_transform([text])
        return vec.get_feature_names_out().tolist()
    except Exception:
        return list({w for w in text.split() if len(w) > 5})[:top_n]


def embed_texts(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    m = SentenceTransformer(model_name)
    return m.encode(texts, show_progress_bar=False)


def build_index(embeddings: np.ndarray):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=min(5, len(embeddings)), metric="cosine")
    idx.fit(embeddings)
    return idx


def generate_document_summary(text: str) -> str:
    sentences = re.split(r"(?<=[.!?]) +", text)
    ob  = sum(1 for s in sentences if any(k in s.lower() for k in ["must", "shall", "required"]))
    pr  = sum(1 for s in sentences if any(k in s.lower() for k in ["must not", "prohibited", "forbidden"]))
    dl  = sum(1 for s in sentences if any(k in s.lower() for k in ["within", "deadline", "by"]))
    pen = sum(1 for s in sentences if any(k in s.lower() for k in ["fine", "penalty", "sanction"]))
    tone = (
        "high compliance risk requiring immediate attention" if pr > 5 or pen > 3
        else "moderate risk requiring periodic review" if ob > 10
        else "low risk with general advisory guidance"
    )
    return (
        f"This document contains {len(sentences)} sentences — "
        f"{ob} obligation(s), {pr} prohibition(s), {dl} deadline reference(s), "
        f"{pen} penalty mention(s). Overall assessment: {tone}."
    )


def kpi_card(col, label: str, value, dot_color: str, sub: str):
    value_color = dot_color if dot_color != "#888" else "inherit"
    col.markdown(
        f"""<div class="kpi-card">
            <div style="font-size:10px;color:gray;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px;">
                <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                    background:{dot_color};margin-right:4px;"></span>{label}
            </div>
            <div style="font-size:28px;font-weight:600;line-height:1;
                color:{value_color};">{value}</div>
            <div style="font-size:10px;color:gray;margin-top:3px;">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def process_documents(docs_dict: dict):
    all_chunks, all_clauses, all_embeddings_map = [], [], {}
    progress = st.progress(0, text="Processing documents…")
    total = len(docs_dict)

    for i, (fname, text) in enumerate(docs_dict.items()):
        progress.progress((i + 1) / total, text=f"Processing: {fname}")
        cleaned = clean_text(text)
        chunks  = chunk_text(cleaned)
        for chunk in chunks:
            all_chunks.append({"source": fname, "text": chunk})
        clauses = extract_compliance_clauses(cleaned)
        for c in clauses:
            c["source"] = fname
        all_clauses.extend(clauses)

    progress.progress(0.8, text="Building vector index…")
    all_texts = [c["text"] for c in all_chunks]
    if all_texts:
        embs  = embed_texts(all_texts)
        index = build_index(embs)
        pos   = 0
        for fname, text in docs_dict.items():
            cleaned = clean_text(text)
            chunks  = chunk_text(cleaned)
            n = len(chunks)
            all_embeddings_map[fname] = (chunks, embs[pos : pos + n])
            pos += n
        st.session_state.faiss_index    = index
        st.session_state.index_metadata = all_chunks

    st.session_state.all_chunks     = all_chunks
    st.session_state.all_clauses    = all_clauses
    st.session_state.documents      = docs_dict
    st.session_state.embeddings_map = all_embeddings_map
    st.session_state.processed      = True
    progress.empty()
    return all_clauses, all_chunks


# CACHED LOADERS


@st.cache_data
def load_data():
    if not CLAUSES_PATH.exists():
        return pd.DataFrame(), np.array([]), [], pd.DataFrame(), pd.DataFrame()

    def safe_read_csv(path):
        try:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    clause_df   = safe_read_csv(CLAUSES_PATH)
    embeddings  = np.load(str(EMBEDDINGS_PATH)) if EMBEDDINGS_PATH.exists() else np.array([])
    conflict_df = safe_read_csv(CONFLICT_PATH)
    gap_df      = safe_read_csv(GAP_PATH)
    clause_ids  = json.load(open(IDS_PATH)) if IDS_PATH.exists() else []

    return clause_df, embeddings, clause_ids, conflict_df, gap_df


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


clause_df, embeddings, clause_ids, conflict_df, gap_df = load_data()
model = load_model()


# SIDEBAR

with st.sidebar:

    st.markdown("### Load Documents")
    use_sample     = st.button("Load Sample Policies", use_container_width=True, type="primary")
    uploaded_files = st.file_uploader(
        "Upload your documents", type=["txt", "pdf", "docx"], accept_multiple_files=True
    )

    if use_sample:
        with st.spinner("Loading sample documents…"):
            docs = load_all_documents(str(SAMPLE_DOCS_PATH))
            if docs:
                process_documents(docs)
                st.success(f"Loaded {len(docs)} sample documents!")
            else:
                st.error("Sample documents not found in app/data/raw/")

    if uploaded_files:
        docs = {}
        for f in uploaded_files:
            suffix = "." + f.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            text = load_document(tmp_path)
            if text.strip():
                docs[f.name] = text
            os.unlink(tmp_path)
        if docs:
            with st.spinner("Processing uploaded documents…"):
                process_documents(docs)
            st.success(f"Processed {len(docs)} documents!")

    if st.session_state.processed:
        st.divider()
        st.markdown("### Loaded Documents")
        for fname in st.session_state.documents:
            st.markdown(f"`{fname}`")
        tc  = len(st.session_state.all_clauses)
        hrc = sum(1 for c in st.session_state.all_clauses if c["risk"] == "High")
        st.markdown(f"- **Clauses found:** {tc}")
        st.markdown(f"- **High risk items:** {hrc}")
        st.markdown(f"- **Chunks indexed:** {len(st.session_state.all_chunks)}")

    st.divider()

    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 12px !important;
        }
        </style>""", unsafe_allow_html=True)
    st.markdown("### Pre-processed Stats")
    if not clause_df.empty and "risk_level" in clause_df.columns:
        st.metric("Total Clauses",  len(clause_df))
        st.metric("🔴 High Risk",   int((clause_df["risk_level"] == "HIGH").sum()))
        st.metric("🟡 Medium Risk", int((clause_df["risk_level"] == "MEDIUM").sum()))
        st.metric("🟢 Low Risk",    int((clause_df["risk_level"] == "LOW").sum()))
    else:
        st.caption("Run Step 1–3 notebooks to generate pre-processed data.")


# MAIN HEADER

st.markdown("""
<div class="main-header">
    <h2 style="margin:0;">AI-Driven Policy & Compliance Intelligence</h2>
    <p style="margin:0.3rem 0 0;opacity:0.85;font-size:0.95rem;">
        Automatic compliance extraction · Risk detection · Semantic Q&A · Document summary
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Home Dashboard",
    "Document Explorer",
    "Compliance Q&A",
    "Risk Dashboard",
    "Document Summary",
])


# TAB 1 — HOME DASHBOARD

with tabs[0]:

    if clause_df.empty or "risk_level" not in clause_df.columns:
        st.info("Load documents from the sidebar, or run the Step 1–3 notebooks first.")
        st.stop()

    total  = len(clause_df)
    high   = int((clause_df["risk_level"] == "HIGH").sum())
    medium = int((clause_df["risk_level"] == "MEDIUM").sum())
    low    = int((clause_df["risk_level"] == "LOW").sum())

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Total clauses",  total,           "#888",    "across all documents")
    kpi_card(k2, "High risk",      high,             "#E24B4A", "require immediate review")
    kpi_card(k3, "Conflicts",      len(conflict_df), "#EF9F27", "clause contradictions")
    kpi_card(k4, "Gaps",           len(gap_df),      "#378ADD", "missing provisions")

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        sel_doc  = st.multiselect("Document",    clause_df["document"].unique(),    default=clause_df["document"].unique())
    with f2:
        sel_risk = st.multiselect("Risk level",  ["HIGH", "MEDIUM", "LOW"],          default=["HIGH", "MEDIUM", "LOW"])
    with f3:
        sel_type = st.multiselect("Clause type", clause_df["clause_type"].unique(), default=clause_df["clause_type"].unique())

    filtered_df = clause_df[
        clause_df["document"].isin(sel_doc) &
        clause_df["risk_level"].isin(sel_risk) &
        clause_df["clause_type"].isin(sel_type)
    ]

    _CL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), font=dict(size=11))

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        st.caption("RISK DISTRIBUTION")
        rc = filtered_df["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk Level", "Count"]
        fig = px.pie(rc, names="Risk Level", values="Count", hole=0.65,
                    color="Risk Level", color_discrete_map=RISK_COLORS)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                        marker=dict(line=dict(color="rgba(0,0,0,0)", width=0)))
        fig.update_layout(
            **_CL, showlegend=False, height=240,
            annotations=[dict(text=f"<b>{len(filtered_df)}</b><br><span style='font-size:10px'>clauses</span>",
                        x=0.5, y=0.5, font_size=20, showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.caption("CLAUSE TYPES")
        tc2 = filtered_df["clause_type"].value_counts().reset_index()
        tc2.columns = ["Clause Type", "Count"]
        fig2 = px.bar(tc2, x="Count", y="Clause Type", orientation="h",
                    color="Count", color_continuous_scale=[[0, "#B5D4F4"], [1, "#185FA5"]])
        fig2.update_layout(**_CL, coloraxis_showscale=False, height=240,
                    yaxis=dict(categoryorder="total ascending", title=""),
                    xaxis=dict(title="", showgrid=True, gridcolor="rgba(136,135,128,.15)"))
        st.plotly_chart(fig2, use_container_width=True)

    with r2c1:
        st.caption("OVERALL RISK SCORE")
        avg_risk = filtered_df["risk_score"].mean() if len(filtered_df) else 0
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number", value=round(avg_risk, 1),
            number={"font": {"size": 36}},
            gauge={
                "axis":    {"range": [0, 100], "tickwidth": 0},
                "bar":     {"color": "#E24B4A", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps":   [
                    {"range": [0, 40],  "color": "#EAF3DE"},
                    {"range": [40, 70], "color": "#FAEEDA"},
                    {"range": [70, 100],"color": "#FCEBEB"},
                ],
            },
        ))
        fig3.update_layout(**_CL, height=240)
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        st.caption("DOCUMENT BREAKDOWN")
        stacked = filtered_df.groupby(["document", "risk_level"]).size().reset_index(name="count")
        fig4 = px.bar(stacked, x="document", y="count", color="risk_level",
                color_discrete_map=RISK_COLORS, barmode="stack")
        fig4.update_layout(
            **_CL, height=240,
            legend=dict(title="", orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(title=""),
            yaxis=dict(title="", showgrid=True, gridcolor="rgba(136,135,128,.15)"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Detailed Data")
    drill    = st.selectbox("Drilldown by Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])
    drill_df = filtered_df if drill == "All" else filtered_df[filtered_df["risk_level"] == drill]
    cols_show = [c for c in ["clause_id", "document", "section", "risk_level", "risk_score"] if c in drill_df.columns]
    st.dataframe(drill_df[cols_show], use_container_width=True)



# TAB 2 — DOCUMENT EXPLORER

with tabs[1]:

    if clause_df.empty or "risk_level" not in clause_df.columns:
        st.info("No pre-processed data found. Run Step 1–3 notebooks first.")
        st.stop()

    documents = clause_df["document"].unique().tolist()
    sel_col, _ = st.columns([1, 2])
    with sel_col:
        selected_doc = st.selectbox("Select document", documents,
                                    label_visibility="collapsed", key="doc_explorer_select")

    doc_clauses = clause_df[clause_df["document"] == selected_doc]
    avg_score   = doc_clauses["risk_score"].mean() if len(doc_clauses) else 0
    high_count  = int((doc_clauses["risk_level"] == "HIGH").sum())

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Total clauses",  len(doc_clauses),                "#888",    "in this document")
    kpi_card(k2, "High risk",      high_count,                      "#E24B4A", "need review")
    kpi_card(k3, "Avg risk score", f"{avg_score:.1f}",              "#EF9F27", "out of 100")
    kpi_card(k4, "Sections",       doc_clauses["section"].nunique(), "#378ADD", "unique sections")

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    cf1, cf2 = st.columns(2)
    with cf1:
        risk_filter = st.multiselect("Filter by Risk Level:", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    with cf2:
        type_filter = st.multiselect("Filter by Clause Type:", doc_clauses["clause_type"].unique().tolist(),
                        default=doc_clauses["clause_type"].unique().tolist())

    filtered = doc_clauses[
        doc_clauses["risk_level"].isin(risk_filter) &
        doc_clauses["clause_type"].isin(type_filter)
    ]
    st.markdown(f"**Showing {len(filtered)} clauses**")
    st.markdown("---")

    for _, row in filtered.iterrows():
        rl    = row["risk_level"].lower()
        badge = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[row["risk_level"]]
        with st.expander(
            f"{badge} {row['clause_id']} | {row['clause_type'].upper()} | Score: {row['risk_score']}"
        ):
            ca, cb = st.columns([3, 1])
            with ca:
                st.markdown(f"**Section:** {row.get('section', 'N/A')}")
                st.markdown("**Clause Text:**")
                st.markdown(f'<div class="risk-{rl}">{row["clause_text"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**Explanation:** {row.get('explanation', 'N/A')}")
            with cb:
                st.metric("Risk Score", f"{row['risk_score']}/100")
                st.metric("Risk Level", row["risk_level"])
                st.metric("Words",      row.get("word_count", "N/A"))
                if str(row.get("has_law_reference", "")).lower() == "true":
                    st.success("Law Reference")
                if str(row.get("has_deadline", "")).lower() == "true":
                    st.warning("Has Deadline")
                if str(row.get("has_penalty", "")).lower() == "true":
                    st.error("Has Penalty")

# TAB 3 — COMPLIANCE Q&A

with tabs[2]:

    st.markdown("## Compliance Q&A")
    st.markdown("Ask any compliance question in plain English. The system retrieves the most semantically relevant clauses.")
    st.markdown("---")

    def answer_question(query: str, cdf: pd.DataFrame, embs: np.ndarray,
                        mdl: SentenceTransformer, top_k: int = 5) -> list:
        if embs is None or len(embs) == 0:
            return []
        q_emb = mdl.encode([query], convert_to_numpy=True)
        sims  = cosine_similarity(q_emb, embs)[0]
        top_i = np.argsort(sims)[::-1][:top_k]
        return [{
            "clause_id":   cdf.iloc[i]["clause_id"],
            "document":    cdf.iloc[i]["document"],
            "section":     cdf.iloc[i].get("section", ""),
            "clause_text": cdf.iloc[i]["clause_text"],
            "clause_type": cdf.iloc[i]["clause_type"],
            "risk_level":  cdf.iloc[i]["risk_level"],
            "risk_score":  cdf.iloc[i]["risk_score"],
            "explanation": cdf.iloc[i].get("explanation", ""),
            "similarity":  round(float(sims[i]), 4),
        } for i in top_i]

    samples = [
        "What are our obligations for data breach notification?",
        "What are the penalties for non-compliance?",
        "Who is responsible for data protection?",
        "What data retention rules apply?",
        "What is prohibited regarding employee data?",
        "What emergency preparedness requirements exist?",
        "What are the radiation dose limits for workers?",
    ]
    sel_sample = st.selectbox("Sample Questions:", ["-- Type your own --"] + samples)
    query  = st.text_input(
        "Your Question:",
        value=sel_sample if sel_sample != "-- Type your own --" else "",
        placeholder="e.g. What must we do in case of a data breach?",
    )
    top_k = st.slider("Number of results:", 3, 10, 5)

    if st.button("Search", type="primary") and query:
        if clause_df.empty or len(embeddings) == 0:
            st.warning("No pre-processed data available. Run Step 1–3 notebooks first.")
        else:
            with st.spinner("Searching policy documents…"):
                results = answer_question(query, clause_df, embeddings, model, top_k)
            st.markdown("---")
            st.markdown(f"### Results for: *'{query}'*")
            st.markdown(f"Found **{len(results)}** relevant clauses")
            st.markdown("---")
            for i, r in enumerate(results, 1):
                badge = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r["risk_level"], "⚪")
                with st.expander(
                    f"#{i} | {badge} {r['clause_id']} | Match: {r['similarity']*100:.1f}% | {r['section']}"
                ):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Document:** {r['document']}")
                        st.markdown(f"**Section:** {r['section']}")
                        st.markdown(f"**Clause Type:** `{r['clause_type'].upper()}`")
                        st.markdown("**Relevant Text:**")
                        rl = r["risk_level"].lower()
                        st.markdown(f'<div class="risk-{rl}">{r["clause_text"]}</div>', unsafe_allow_html=True)
                        if r["explanation"]:
                            st.info(f"💡 **Risk Explanation:** {r['explanation']}")
                    with c2:
                        st.metric("Relevance",  f"{r['similarity']*100:.1f}%")
                        st.metric("Risk Score", f"{r['risk_score']}/100")
                        st.metric("Risk Level", r["risk_level"])
            st.session_state.chat_history.append(query)

    if st.button("Clear History"):
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Search History")
        for q in st.session_state.chat_history[-5:]:
            st.markdown(f"- {q}")

# TAB 4 — RISK DASHBOARD

with tabs[3]:

    if clause_df.empty or "risk_level" not in clause_df.columns:
        st.info("No pre-processed data found. Run Step 1–3 notebooks first.")
        st.stop()

    df_risk = clause_df.copy()
    df_risk["risk_score"]  = pd.to_numeric(df_risk["risk_score"],  errors="coerce")
    df_risk["risk_level"]  = df_risk["risk_level"].fillna("LOW")
    df_risk["clause_type"] = df_risk["clause_type"].fillna("Unknown")
    df_risk["document"]    = df_risk["document"].fillna("Unknown")
    df_risk = df_risk.dropna(subset=["risk_score"])

    _CL2 = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10, l=10, r=10), font=dict(size=11))

    rt1, rt2, rt3 = st.tabs(["Risk Overview", "Conflict Registry", "Gap Analysis"])

    with rt1:
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, label, val in [
            (k1, "High",   int((df_risk["risk_level"] == "HIGH").sum())),
            (k2, "Medium", int((df_risk["risk_level"] == "MEDIUM").sum())),
            (k3, "Low",    int((df_risk["risk_level"] == "LOW").sum())),
            (k4, "Avg",    f"{df_risk['risk_score'].mean():.1f}"),
            (k5, "Max",    int(df_risk["risk_score"].max()) if not df_risk.empty else 0),
        ]:
            col.metric(label, val)

        st.subheader("Score Distribution")
        fig = px.histogram(df_risk, x="risk_score", color="risk_level", color_discrete_map=RISK_COLORS)
        fig.update_layout(**_CL2)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk by Clause Type")
        tr = df_risk.groupby("clause_type")["risk_score"].mean().reset_index()
        fig2 = px.bar(tr, x="risk_score", y="clause_type", orientation="h",color="risk_score", color_continuous_scale="RdYlGn_r")
        fig2.update_layout(**_CL2)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("High Risk Clauses")
        hr_df = df_risk[df_risk["risk_level"] == "HIGH"]
        if hr_df.empty:
            st.info("No high risk clauses found.")
        else:
            cols_show = [c for c in ["clause_id", "document", "section", "risk_score", "clause_type"] if c in hr_df.columns]
            st.dataframe(hr_df[cols_show], use_container_width=True)
            st.download_button(
                "⬇Download High Risk Clauses",
                data=hr_df.to_csv(index=False),
                file_name="high_risk_clauses.csv",
                mime="text/csv",
            )

    with rt2:
        if conflict_df.empty:
            st.info("No conflicts detected. Run Step 2 notebook to generate conflict registry.")
        else:
            st.subheader("Conflict Registry")
            if "conflict_type" in conflict_df.columns:
                cc = conflict_df["conflict_type"].value_counts().reset_index()
                cc.columns = ["Conflict Type", "Count"]
                fig = px.bar(cc, x="Conflict Type", y="Count", color="Conflict Type")
                st.plotly_chart(fig, use_container_width=True)

            show_df = conflict_df
            if "conflict_severity" in conflict_df.columns:
                sev = st.multiselect(
                    "Filter by Severity:",
                    conflict_df["conflict_severity"].unique().tolist(),
                    default=conflict_df["conflict_severity"].unique().tolist(),
                )
                show_df = conflict_df[conflict_df["conflict_severity"].isin(sev)]

            for _, row in show_df.iterrows():
                ctype = row.get("conflict_type", "Unknown")
                try:    sim_str = f"{float(row.get('similarity', '')):.2f}"
                except: sim_str = "—"
                with st.expander(f"{ctype} | Similarity: {sim_str}"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Clause A:** `{row.get('clause_id_a', '')}`")
                    c1.markdown(f"**Doc:** {row.get('document_a', '')}")
                    c1.markdown(str(row.get("text_a", ""))[:200])
                    c2.markdown(f"**Clause B:** `{row.get('clause_id_b', '')}`")
                    c2.markdown(f"**Doc:** {row.get('document_b', '')}")
                    c2.markdown(str(row.get("text_b", ""))[:200])

    with rt3:
        if gap_df.empty:
            st.success("No compliance gaps detected.")
        else:
            st.subheader("Gap Analysis")
            st.error(f"⚠️ {len(gap_df)} compliance gap(s) found.")
            for _, row in gap_df.iterrows():
                badge = "🔴" if row.get("gap_severity") == "HIGH" else "🟡"
                with st.expander(f"{badge} {row.get('clause_id', '')} | {row.get('section', '')}"):
                    st.markdown(f"**Document:** {row.get('document', '')}")
                    st.markdown(f"**Obligation:** {row.get('obligation_text', '')}")
                    st.markdown(f"**Severity:** {row.get('gap_severity', '')}")
                    st.warning("No matching control clause found. Policy update may be required.")

# TAB 5 — DOCUMENT SUMMARY

with tabs[4]:

    st.markdown("### Document Summary")
    st.markdown("Select a document to generate a full summary with risk profile, obligations, and recommendations.")
    st.divider()

    summary_source = st.radio(
        "Summarise from:",
        ["Pre-processed data (notebooks)", "Uploaded / Sample documents"],
        horizontal=True,
    )

    if summary_source == "Pre-processed data (notebooks)":
        if clause_df.empty or "risk_level" not in clause_df.columns:
            st.info("No pre-processed data found. Run Step 1–3 notebooks first.")
            st.stop()
        doc_options = clause_df["document"].unique().tolist()
    else:
        if not st.session_state.processed:
            st.info("No documents loaded. Use the sidebar to load or upload documents first.")
            st.stop()
        doc_options = list(st.session_state.documents.keys())

    selected_doc_sum = st.selectbox(
        "Select a document:",
        ["-- Select a document --"] + doc_options,
        key="summary_doc_select",
    )
    generate_btn = st.button("Generate Summary", type="primary", key="gen_sum_btn")

    if generate_btn and selected_doc_sum == "-- Select a document --":
        st.warning("Please select a document first.")

    if generate_btn and selected_doc_sum != "-- Select a document --":

        # ── Animated loader ───────────────────────────────────────────────────
        st.markdown("---")
        pbar   = st.progress(0)
        stxt   = st.empty()
        sdispl = st.empty()
        steps  = [
            "Reading and cleaning document text…",
            "Segmenting document into clauses…",
            "Running Named Entity Recognition…",
            "Scoring risk levels per clause…",
            "Generating AI summary and recommendations…",
        ]
        for i, step in enumerate(steps):
            stxt.markdown(f"**{step}**")
            cl = "".join(
                f"{ '✅' if j < i else '🔄' if j == i else '⬜' } {s}\n\n"
                for j, s in enumerate(steps)
            )
            sdispl.markdown(cl)
            pbar.progress(int(((i + 1) / len(steps)) * 100))
            time.sleep(0.4)

        pbar.progress(100)
        stxt.empty()
        sdispl.empty()
        pbar.empty()
        st.success(f"Summary generated for **{selected_doc_sum}**")
        st.divider()

        # ── Pull data ─────────────────────────────────────────────────────────
        if summary_source == "Pre-processed data (notebooks)":
            ddf    = clause_df[clause_df["document"] == selected_doc_sum]
            total  = len(ddf)
            high_c = int((ddf["risk_level"] == "HIGH").sum())
            med_c  = int((ddf["risk_level"] == "MEDIUM").sum())
            low_c  = int((ddf["risk_level"] == "LOW").sum())
            ob_c   = int((ddf["clause_type"] == "obligation").sum())
            pr_c   = int((ddf["clause_type"] == "prohibition").sum())
            dl_c   = int((ddf["clause_type"] == "condition").sum())
            raw_text        = ""
            session_clauses = None
        else:
            raw_text        = st.session_state.documents.get(selected_doc_sum, "")
            session_clauses = [c for c in st.session_state.all_clauses if c.get("source") == selected_doc_sum]
            ddf    = None
            total  = len(session_clauses)
            high_c = sum(1 for c in session_clauses if c["risk"] == "High")
            med_c  = sum(1 for c in session_clauses if c["risk"] == "Medium")
            low_c  = sum(1 for c in session_clauses if c["risk"] == "Low")
            ob_c   = sum(1 for c in session_clauses if c["type"] == "obligation")
            pr_c   = sum(1 for c in session_clauses if c["type"] == "prohibition")
            dl_c   = sum(1 for c in session_clauses if c["type"] == "deadline")

        # ── Header metrics ─────────────────────────────────────────────────────
        st.markdown(f"#### {selected_doc_sum}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Clauses", total)
        m2.metric("High Risk",     high_c)
        m3.metric("Obligations",   ob_c)
        m4.metric("Prohibitions",  pr_c)
        st.divider()

        s1, s2, s3, s4 = st.tabs(["Executive Summary", "Risk Profile", "Clause Breakdown", "Recommendations"])

        with s1:
            st.markdown("#### Executive Summary")
            if raw_text:
                ai_sum = generate_document_summary(raw_text)
                kws    = extract_keywords(raw_text, top_n=10)
            else:
                tone = (
                    "high compliance risk" if high_c > total * 0.3
                    else "moderate risk"   if high_c > 0
                    else "low risk"
                )
                ai_sum = (
                    f"This document contains {total} clauses — {ob_c} obligation(s), "
                    f"{pr_c} prohibition(s), {dl_c} condition(s). "
                    f"Overall assessment: {tone}. "
                    f"{high_c} clause(s) require priority action."
                )
                kws = []
            st.markdown(f'<div class="summary-header">{ai_sum}</div>', unsafe_allow_html=True)
            if kws:
                st.markdown("**Key terms:** " + " · ".join([f"`{k}`" for k in kws]))

        with s2:
            st.markdown("#### Risk Profile")
            r1, r2, r3 = st.columns(3)
            r1.metric("High Risk",   high_c, f"{round(high_c / max(total, 1) * 100)}%")
            r2.metric("Medium Risk", med_c,  f"{round(med_c  / max(total, 1) * 100)}%")
            r3.metric("Low Risk",    low_c,  f"{round(low_c  / max(total, 1) * 100)}%")
            st.divider()
            if ddf is not None and not ddf.empty:
                rc2 = (ddf["risk_level"].value_counts()
                    .reset_index()
                    .rename(columns={"risk_level": "Risk Level", "count": "Count"}))
                fig = px.pie(rc2, names="Risk Level", values="Count",
                        color="Risk Level", color_discrete_map=RISK_COLORS, hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**High risk clauses:**")
                hr = ddf[ddf["risk_level"] == "HIGH"]
                if hr.empty:
                    st.success("No high risk clauses.")
                else:
                    for _, row in hr.head(5).iterrows():
                        st.markdown(
                            f'<div class="risk-high"><b>{row["clause_type"].upper()}</b>'
                            f'<br>{row["clause_text"][:300]}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                hr = [c for c in (session_clauses or []) if c["risk"] == "High"]
                if not hr:
                    st.success("No high risk clauses.")
                else:
                    for c in hr[:5]:
                        st.markdown(
                            f'<div class="risk-high"><b>{c["type"].upper()}</b>'
                            f'<br>{c["text"][:300]}</div>',
                            unsafe_allow_html=True,
                        )

        with s3:
            st.markdown("#### Clause Breakdown")
            if ddf is not None and not ddf.empty:
                tf = st.multiselect("Type:", ddf["clause_type"].unique().tolist(),
                                    default=ddf["clause_type"].unique().tolist(), key="s3tf")
                rf = st.multiselect("Risk:", ["HIGH", "MEDIUM", "LOW"],
                                    default=["HIGH", "MEDIUM", "LOW"], key="s3rf")
                fcls = ddf[ddf["clause_type"].isin(tf) & ddf["risk_level"].isin(rf)]
                st.markdown(f"Showing **{len(fcls)}** clauses")
                st.divider()
                for i, (_, row) in enumerate(fcls.iterrows(), 1):
                    ri  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[row["risk_level"]]
                    rl  = row["risk_level"].lower()
                    st.markdown(
                        f'<div class="risk-{rl}"><b>{i}. {row["clause_type"].upper()}</b> {ri}'
                        f'<br><span style="font-size:0.9rem;">{row["clause_text"][:350]}</span></div>',
                        unsafe_allow_html=True,
                    )
                if not fcls.empty:
                    st.divider()
                    st.download_button(
                        "⬇️ Download clauses as CSV",
                        data=fcls.to_csv(index=False),
                        file_name=f"clauses_{selected_doc_sum}.csv",
                        mime="text/csv",
                    )
            else:
                tf = st.multiselect(
                    "Type:", ["obligation", "prohibition", "deadline", "general"],
                    default=["obligation", "prohibition", "deadline", "general"], key="s3tf_s",
                )
                rf = st.multiselect("Risk:", ["High", "Medium", "Low"],
                                    default=["High", "Medium", "Low"], key="s3rf_s")
                fcls_s = [c for c in (session_clauses or []) if c["type"] in tf and c["risk"] in rf]
                st.markdown(f"Showing **{len(fcls_s)}** clauses")
                st.divider()
                for i, c in enumerate(fcls_s, 1):
                    ri  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(c["risk"], "⚪")
                    rl2 = c["risk"].lower()
                    st.markdown(
                        f'<div class="risk-{rl2}"><b>{i}. {c["type"].upper()}</b> {ri}'
                        f'<br><span style="font-size:0.9rem;">{c["text"][:350]}</span></div>',
                        unsafe_allow_html=True,
                    )

        with s4:
            st.markdown("#### Recommendations")
            st.caption("Auto-generated action items based on risk levels.")

            if ddf is not None and not ddf.empty:
                hi_items = ddf[ddf["risk_level"] == "HIGH"]
                me_items = ddf[ddf["risk_level"] == "MEDIUM"]
                is_df    = True
            else:
                hi_items = [c for c in (session_clauses or []) if c["risk"] == "High"]
                me_items = [c for c in (session_clauses or []) if c["risk"] == "Medium"]
                is_df    = False

            empty_hi = hi_items.empty if is_df else not hi_items
            empty_me = me_items.empty if is_df else not me_items

            if empty_hi:
                st.success("No high risk items found.")
            else:
                st.markdown("**🔴 Immediate action required:**")
                items = hi_items.iterrows() if is_df else enumerate(hi_items)
                for _, row in items:
                    txt   = row["clause_text"] if is_df else row["text"]
                    ctype = row["clause_type"]  if is_df else row["type"]
                    with st.expander(f"{ctype.upper()} — {txt[:70]}…"):
                        st.markdown(f'<div class="risk-high">{txt}</div>', unsafe_allow_html=True)
                        st.markdown("**Action:** Review immediately. Assign responsible owner. Verify compliance.")

            if empty_me:
                st.info("No medium risk items found.")
            else:
                st.markdown("**🟡 Review recommended:**")
                items = me_items.iterrows() if is_df else enumerate(me_items)
                for _, row in items:
                    txt   = row["clause_text"] if is_df else row["text"]
                    ctype = row["clause_type"]  if is_df else row["type"]
                    with st.expander(f"{ctype.upper()} — {txt[:70]}…"):
                        st.markdown(f'<div class="risk-medium">{txt}</div>', unsafe_allow_html=True)
                        st.markdown("**Action:** Verify implementation. Include in next review cycle.")

            st.divider()
            if is_df:
                frames = [f for f in [
                    hi_items if not empty_hi else None,
                    me_items if not empty_me else None,
                ] if f is not None]
                if frames:
                    rec = pd.concat(frames)
                    rec = rec.copy()
                    rec["Action"] = rec["risk_level"].map(
                        {"HIGH": "Immediate review", "MEDIUM": "Next review cycle"}
                    )
                    cols_dl = [c for c in ["clause_id", "risk_level", "clause_type", "clause_text", "Action"] if c in rec.columns]
                    st.download_button(
                        "Download recommendations as CSV",
                        data=rec[cols_dl].to_csv(index=False),
                        file_name=f"recommendations_{selected_doc_sum}.csv",
                        mime="text/csv",
                    )
