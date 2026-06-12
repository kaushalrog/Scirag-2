"""
frontend/app.py
----------------
Streamlit chat interface for SciRAG-UQ.

Features
--------
- Chat interface with streaming support
- Confidence badge on every answer
- Source card panel with arXiv links
- UQ breakdown radar chart (Plotly)
- Corpus stats sidebar
- PDF upload widget
"""

import requests
import streamlit as st
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SciRAG-UQ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.confidence-high   { background:#22c55e; color:white; padding:3px 10px; border-radius:12px; font-weight:600; }
.confidence-medium { background:#f59e0b; color:white; padding:3px 10px; border-radius:12px; font-weight:600; }
.confidence-low    { background:#ef4444; color:white; padding:3px 10px; border-radius:12px; font-weight:600; }
.confidence-vlow   { background:#7c3aed; color:white; padding:3px 10px; border-radius:12px; font-weight:600; }
.source-card { border:1px solid #e5e7eb; border-radius:8px; padding:10px 14px; margin:6px 0; background:#f9fafb; }
.abstain-banner { background:#fef3c7; border-left:4px solid #f59e0b; padding:12px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_health():
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return None

def get_sources():
    try:
        return requests.get(f"{API_BASE}/sources", timeout=5).json()
    except Exception:
        return {"sources": [], "total": 0}

def ask_question(question: str, use_cot: bool) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "use_cot": use_cot},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def ingest_papers(query: str, max_papers: int) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/ingest",
            json={"query": query, "max_papers": max_papers, "enrich_s2": True},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Ingest error: {e}")
        return None

def confidence_badge(label: str, score: float) -> str:
    cls = {
        "HIGH": "confidence-high",
        "MEDIUM": "confidence-medium",
        "LOW": "confidence-low",
        "VERY LOW": "confidence-vlow",
    }.get(label, "confidence-low")
    return f'<span class="{cls}">{label} {score:.0%}</span>'

def uq_radar(metrics: dict):
    keys = ["retrieval_confidence", "generation_confidence",
            "semantic_consistency", "retrieval_coverage"]
    labels = ["Retrieval Conf.", "Generation Conf.", "Consistency", "Coverage"]
    values = [metrics.get(k, 0) for k in keys]
    values.append(values[0])  # close polygon

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(59,130,246,0.2)",
        line=dict(color="rgb(59,130,246)", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/SciRAG--UQ-BDA2026-blue", width=160)
    st.title("SciRAG-UQ")
    st.caption("Uncertainty-Aware Scientific RAG")
    st.divider()

    health = get_health()
    if health:
        st.success(f"API online — model: `{health.get('model','')}`")
        st.metric("Corpus size", f"{health.get('corpus_size', 0):,} chunks")
    else:
        st.error("API offline — start the FastAPI server")

    st.divider()
    st.subheader("⚙️ Settings")
    use_cot = st.toggle("Chain-of-Thought", value=False, help="Slower but more reasoned")

    st.divider()
    st.subheader("📥 Ingest Papers")
    ingest_query = st.text_input("arXiv query", placeholder="RAG retrieval augmented generation")
    max_papers = st.slider("Max papers", 5, 50, 20)
    if st.button("Ingest", use_container_width=True):
        with st.spinner("Ingesting…"):
            result = ingest_papers(ingest_query, max_papers)
            if result:
                st.success(f"Ingested {result['ingested']} papers")

    st.divider()
    st.subheader("📄 Upload PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Upload & Ingest"):
        with st.spinner("Uploading…"):
            resp = requests.post(
                f"{API_BASE}/ingest/pdf",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
            if resp.ok:
                d = resp.json()
                st.success(f"Ingested: {d['chunks']} chunks")
            else:
                st.error("Upload failed")

    st.divider()
    sources_data = get_sources()
    with st.expander(f"📚 Corpus ({sources_data['total']} chunks)"):
        for s in sources_data.get("sources", [])[:20]:
            st.caption(f"• {s}")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("🔬 SciRAG-UQ")
st.caption("Ask questions about the ingested scientific literature corpus")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            if "meta" in msg:
                meta = msg["meta"]
                col1, col2 = st.columns([2, 1])
                with col1:
                    badge = confidence_badge(meta["confidence_label"], meta["confidence"])
                    st.markdown(badge, unsafe_allow_html=True)
                    if meta.get("abstained"):
                        st.markdown(
                            f'<div class="abstain-banner">⚠️ {meta["abstention_reason"]}</div>',
                            unsafe_allow_html=True,
                        )
                    if meta.get("sources"):
                        st.markdown("**Sources**")
                        for src in meta["sources"]:
                            url = src.get("url", "#")
                            title = src.get("title", "Unknown")
                            score = src.get("score", 0)
                            st.markdown(
                                f'<div class="source-card">'
                                f'<a href="{url}" target="_blank"><b>{title[:60]}</b></a>'
                                f' — score: <b>{score:.3f}</b>'
                                f'<br><small>§ {src.get("section","—")} | {src.get("published","")}</small>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                with col2:
                    if meta.get("uncertainty_breakdown"):
                        st.plotly_chart(
                            uq_radar(meta["uncertainty_breakdown"]),
                            use_container_width=True,
                        )
        else:
            st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a question about the scientific corpus…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            response = ask_question(prompt, use_cot=use_cot)

        if response:
            st.markdown(response["answer"])
            badge = confidence_badge(response["confidence_label"], response["confidence"])
            st.markdown(badge, unsafe_allow_html=True)

            if response.get("abstained"):
                st.markdown(
                    f'<div class="abstain-banner">⚠️ {response["abstention_reason"]}</div>',
                    unsafe_allow_html=True,
                )

            col1, col2 = st.columns([2, 1])
            with col1:
                if response.get("sources"):
                    st.markdown("**Sources**")
                    for src in response["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<a href="{src.get("url","#")}" target="_blank">'
                            f'<b>{src.get("title","?")[:60]}</b></a>'
                            f' — score: <b>{src.get("score",0):.3f}</b>'
                            f'<br><small>§ {src.get("section","—")}</small>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            with col2:
                if response.get("uncertainty_breakdown"):
                    st.plotly_chart(
                        uq_radar(response["uncertainty_breakdown"]),
                        use_container_width=True,
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "meta": response,
            })
