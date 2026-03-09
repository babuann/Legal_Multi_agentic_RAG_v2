import logging
import re
import sys
from typing import List

import streamlit as st
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
for _lib in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

st.set_page_config(
    page_title="Multi-Agent RAG | Legal Documents",
    page_icon="⚖️",
    layout="wide",
)

from src.config import settings  # noqa: E402
from src.deep_agents_flow import DeepAgentState, run_deep_agents_flow  # noqa: E402
from src.langgraph_flow import run_langgraph_flow  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    return VectorStore()


def _source_table(docs: List[Document]) -> None:
    if not docs:
        st.caption("No sources retrieved.")
        return

    rows = []
    for i, doc in enumerate(docs, start=1):
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.replace("\n", " ").strip()[:200]
        rows.append({"#": f"[{i}]", "Document": src, "Page": page, "Snippet": snippet + "…"})

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#":        st.column_config.TextColumn(width="small"),
            "Document": st.column_config.TextColumn(width="medium"),
            "Page":     st.column_config.TextColumn(width="small"),
            "Snippet":  st.column_config.TextColumn(width="large"),
        },
    )


def _highlight_citations(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r"**[\1]**", text)


def _render_langgraph(result: dict) -> None:
    val     = result.get("validation", {})
    is_valid = result.get("is_valid", False)
    retries = max(result.get("retry_count", 1) - 1, 0)
    docs: List[Document] = result.get("retrieved_docs", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route",         result.get("route", "—"))
    c2.metric("Valid",         "Yes" if is_valid else "No")
    c3.metric("Retries",       retries)
    c4.metric("Overall score", f"{val.get('overall_score', 0):.2f}")

    with st.expander("Validation scores", expanded=False):
        s1, s2, s3 = st.columns(3)
        s1.metric("Relevance",    f"{val.get('relevance',    0):.2f}")
        s2.metric("Grounding",    f"{val.get('grounding',    0):.2f}")
        s3.metric("Completeness", f"{val.get('completeness', 0):.2f}")
        if val.get("feedback"):
            st.info(f"Validator feedback: {val['feedback']}")
        if not is_valid and val.get("refined_query"):
            st.caption(f"Refined query used on retry: _{val['refined_query']}_")

    st.markdown("#### Answer")
    answer = result.get("answer", "_No answer generated._")
    st.markdown(_highlight_citations(answer))

    st.markdown("#### Source References")
    st.caption(
        "Numbers in the answer (e.g. **[1]**) correspond to the rows below."
    )
    with st.expander(f"Show {len(docs)} retrieved chunk(s)", expanded=True):
        _source_table(docs)


def _render_deepagents(state: DeepAgentState) -> None:
    val = state.validation

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iterations",    state.iterations_completed)
    c2.metric("Research notes", len(state.research_notes))
    c3.metric("Confidence",    f"{state.overall_confidence:.2f}")
    c4.metric("Overall score", f"{val.overall_score:.2f}" if val else "—")

    with st.expander("Research log (sub-questions & sources)", expanded=False):
        for i, note in enumerate(state.research_notes, start=1):
            st.markdown(f"**Note {i}** — _{note.sub_question}_")

            badge_cols = st.columns([1, 1, 4])
            badge_cols[0].caption(f"Route: `{note.route}`")
            badge_cols[1].caption(f"Confidence: `{note.confidence:.2f}`")

            src_tags = "  ".join(f"`{s}`" for s in note.sources)
            badge_cols[2].caption(f"Sources: {src_tags}")

            st.markdown(_highlight_citations(note.answer))

            if i < len(state.research_notes):
                st.divider()

    if val:
        with st.expander("Validation scores", expanded=False):
            v1, v2, v3 = st.columns(3)
            v1.metric("Relevance",    f"{val.relevance:.2f}")
            v2.metric("Grounding",    f"{val.grounding:.2f}")
            v3.metric("Completeness", f"{val.completeness:.2f}")
            if val.feedback:
                st.info(f"Validator feedback: {val.feedback}")

    st.markdown("#### Final Answer")
    st.markdown(_highlight_citations(state.final_answer or "_No answer generated._"))

    st.markdown("#### Documents Used")
    all_sources = []
    for note in state.research_notes:
        for s in note.sources:
            if s not in all_sources:
                all_sources.append(s)

    if all_sources:
        for s in all_sources:
            st.markdown(f"- `{s}`")
    else:
        st.caption("No source metadata available.")


with st.sidebar:
    st.title("RAG Pipeline")
    st.caption("Securities-law multi-agent system")
    st.divider()

    st.subheader("Settings")
    st.markdown(f"**Model:** `{settings.llm_model}`")
    st.markdown(f"**Embeddings:** `{settings.embedding_model.split('/')[-1]}`")
    st.markdown(f"**Chunk size:** {settings.chunk_size} / overlap {settings.chunk_overlap}")
    st.markdown(f"**Rate limit:** {settings.max_requests_per_minute} req/min")
    st.markdown(f"**Max iterations:** {settings.max_iterations} (DeepAgents)")

    st.divider()
    st.subheader("Quick queries")
    preset_queries = {
        "Facebook v. Amalgamated Bank — core issue": (
            "What is the central legal issue in Facebook v. Amalgamated Bank, "
            "and what did the Ninth Circuit decide?"
        ),
        "Petitioner vs Amicus — materiality argument": (
            "How do the petitioner's brief and the US government's amicus brief "
            "differ in their arguments about the materiality standard under "
            "Section 10(b) and Rule 10b-5?"
        ),
        "SEC complaint — key allegations": (
            "What are the main allegations in the SEC complaint and what "
            "securities laws are cited?"
        ),
        "Rule 10b-5 standard — District Court": (
            "How does the District Court in Last Atlantis Capital v. AGS "
            "apply the Rule 10b-5 standard for market manipulation?"
        ),
    }
    selected_preset = st.selectbox(
        "Load a preset query",
        options=["— choose —"] + list(preset_queries.keys()),
    )

    st.divider()
    st.caption("Built with LangChain · LangGraph · ChromaDB · Gemini")


st.title("Multi-Agent RAG — Legal Document Q&A")
st.markdown(
    "Ask a question about the **Facebook v. Amalgamated Bank** Supreme Court "
    "briefs, the **SEC complaint**, and related securities-law opinions. "
    "Inline citations **[N]** in every answer link to the source table below it."
)

default_query = (
    preset_queries[selected_preset]
    if selected_preset != "— choose —"
    else ""
)
query = st.text_area(
    "Your question",
    value=default_query,
    height=100,
    placeholder="e.g. What is the holding in Facebook v. Amalgamated Bank?",
)

flow_choice = st.radio(
    "Which flow to run?",
    options=["Both", "LangGraph only", "DeepAgents only"],
    horizontal=True,
)

run_button = st.button("Run pipeline", type="primary", disabled=not query.strip())

try:
    with st.spinner("Loading vector store …"):
        vector_store = get_vector_store()
except RuntimeError as e:
    st.error(
        f"**Vector store not initialised.**\n\n{e}\n\n"
        "Run `docker compose run --rm ingest` (or `python ingest.py` locally) "
        "then restart the app."
    )
    st.stop()

if run_button and query.strip():
    run_lg = flow_choice in ("Both", "LangGraph only")
    run_da = flow_choice in ("Both", "DeepAgents only")

    lg_col, da_col = (st.columns(2) if flow_choice == "Both"
                      else (st.container(), st.container()))

    if run_lg:
        with lg_col:
            st.subheader("LangGraph Flow")
            st.caption("Formal StateGraph · typed state · validator-driven retry")
            with st.spinner("Running LangGraph flow …"):
                try:
                    lg_result = run_langgraph_flow(query.strip(), vector_store)
                    _render_langgraph(lg_result)
                except Exception as exc:
                    st.error(f"LangGraph flow error: {exc}")

    if run_da:
        with da_col:
            st.subheader("DeepAgents Flow")
            st.caption("Autonomous research loop · decompose → reflect → deepen → synthesise")
            with st.spinner("Running DeepAgents flow (may take ~1 min) …"):
                try:
                    da_state = run_deep_agents_flow(query.strip(), vector_store)
                    _render_deepagents(da_state)
                except Exception as exc:
                    st.error(f"DeepAgents flow error: {exc}")
