import logging
import sys
import textwrap
from typing import Any, Dict

from src.config import settings
from src.deep_agents_flow import DeepAgentState, run_deep_agents_flow
from src.langgraph_flow import run_langgraph_flow
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

for _noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

DEMO_QUERIES = [
    (
        "What is the central legal issue in Facebook v. Amalgamated Bank, "
        "and what did the Ninth Circuit decide?"
    ),
    (
        "How do the petitioner's brief and the US government's amicus brief "
        "differ in their arguments about the materiality standard under "
        "Section 10(b) and Rule 10b-5?"
    ),
]

_SEP = "─" * 72


def _banner(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _print_langgraph_result(result: Dict[str, Any], query: str) -> None:
    _banner("LANGGRAPH FLOW — RESULT")
    print(f"Query   : {query}")
    print(f"Route   : {result['route']}")
    val = result.get("validation", {})
    print(
        f"Scores  : relevance={val.get('relevance', '?'):.2f}  "
        f"grounding={val.get('grounding', '?'):.2f}  "
        f"completeness={val.get('completeness', '?'):.2f}  "
        f"(overall={val.get('overall_score', '?'):.2f})"
    )
    print(f"Valid   : {result['is_valid']}  |  Retries: {result['retry_count'] - 1}")
    if val.get("feedback"):
        print(f"Feedback: {val['feedback']}")
    print("\nAnswer:\n")
    print(textwrap.fill(result["answer"], width=80, subsequent_indent="  "))


def _print_deepagents_result(state: DeepAgentState) -> None:
    _banner("DEEPAGENTS FLOW — RESULT")
    print(f"Query      : {state.original_query}")
    print(f"Iterations : {state.iterations_completed}")
    print(f"Notes      : {len(state.research_notes)} research notes")
    print(f"Confidence : {state.overall_confidence:.2f}")

    if state.validation:
        v = state.validation
        print(
            f"Validation : relevance={v.relevance:.2f}  "
            f"grounding={v.grounding:.2f}  "
            f"completeness={v.completeness:.2f}  "
            f"(overall={v.overall_score:.2f})"
        )

    print("\nResearch Notes Summary:")
    for i, note in enumerate(state.research_notes, start=1):
        print(f"\n  [Note {i}] Sub-question: {note.sub_question}")
        print(f"            Route: {note.route} | Confidence: {note.confidence:.2f}")
        print(f"            Sources: {', '.join(note.sources)}")

    print("\nFinal Answer:\n")
    print(textwrap.fill(state.final_answer, width=80, subsequent_indent="  "))


def main() -> None:
    _banner("Multi-Agent RAG Pipeline — Initialising")
    logger.info("Model         : %s", settings.llm_model)
    logger.info("Embeddings    : %s", settings.embedding_model)
    logger.info("Chunk size    : %d / overlap %d", settings.chunk_size, settings.chunk_overlap)
    logger.info("Rate limit    : %d req/min", settings.max_requests_per_minute)
    logger.info("Max iterations: %d (DeepAgents)", settings.max_iterations)
    logger.info("Data dir      : %s", settings.data_dir)

    _banner("Step 1 — Loading Vector Store")
    vector_store = VectorStore()

    for i, query in enumerate(DEMO_QUERIES, start=1):
        _banner(f"Query {i}/{len(DEMO_QUERIES)}")
        print(f"  {query}\n")

        _banner(f"Query {i} — LangGraph Flow")
        try:
            lg_result = run_langgraph_flow(query, vector_store)
            _print_langgraph_result(lg_result, query)
        except Exception as exc:
            logger.error("LangGraph flow failed: %s", exc, exc_info=True)

        _banner(f"Query {i} — DeepAgents Flow")
        try:
            da_state = run_deep_agents_flow(query, vector_store)
            _print_deepagents_result(da_state)
        except Exception as exc:
            logger.error("DeepAgents flow failed: %s", exc, exc_info=True)

    _banner("Pipeline complete")


if __name__ == "__main__":
    main()
