import logging
from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.agents import (
    RetrieverAgent,
    RouterAgent,
    SynthesizerAgent,
    ValidatorAgent,
    ValidationResult,
)
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class RAGState(TypedDict):
    query: str
    original_query: str
    route: str
    retrieved_docs: List[Document]
    context: str
    answer: str
    validation: Dict[str, Any]
    is_valid: bool
    retry_count: int


def make_route_query_node(router: RouterAgent):
    def route_query(state: RAGState) -> Dict[str, Any]:
        route = router.route(state["query"])
        logger.info("[LangGraph] route_query → %s", route)
        return {"route": route}

    return route_query


def make_synthesize_node(synthesizer: SynthesizerAgent):
    def synthesize(state: RAGState) -> Dict[str, Any]:
        answer = synthesizer.synthesize(
            state["original_query"],
            state["retrieved_docs"],
            state["route"],
        )
        logger.info("[LangGraph] synthesize → %d chars", len(answer))
        return {"answer": answer}

    return synthesize


def make_validate_node(validator: ValidatorAgent):
    def validate(state: RAGState) -> Dict[str, Any]:
        result: ValidationResult = validator.validate(
            state["original_query"],
            state["answer"],
            state["retrieved_docs"],
        )
        logger.info(
            "[LangGraph] validate → valid=%s, score=%.2f",
            result.is_valid,
            result.overall_score,
        )
        return {
            "is_valid": result.is_valid,
            "validation": {
                "relevance": result.relevance,
                "grounding": result.grounding,
                "completeness": result.completeness,
                "overall_score": result.overall_score,
                "feedback": result.feedback,
                "refined_query": result.refined_query,
            },
        }

    return validate


def should_retry(state: RAGState) -> str:
    if state["is_valid"] or state["retry_count"] >= MAX_RETRIES:
        reason = "accepted" if state["is_valid"] else "max retries reached"
        logger.info("[LangGraph] Routing → END (%s)", reason)
        return END

    logger.info("[LangGraph] Routing → retrieve_docs (retry %d)", state["retry_count"] + 1)
    return "retrieve_docs"


def build_langgraph_flow(vector_store: VectorStore) -> Any:
    router = RouterAgent()
    retriever = RetrieverAgent(vector_store)
    synthesizer = SynthesizerAgent()
    validator = ValidatorAgent()

    def retrieve_docs_with_retry_tracking(state: RAGState) -> Dict[str, Any]:
        active_query = state["query"]
        retry = state["retry_count"]

        if retry > 0:
            refined = state["validation"].get("refined_query", active_query)
            if refined and refined != active_query:
                logger.info("[LangGraph] Using refined query: %s", refined[:80])
                active_query = refined

        docs = retriever.retrieve(active_query, state["route"])
        context = RetrieverAgent.format_context(docs)

        return {
            "query": active_query,
            "retrieved_docs": docs,
            "context": context,
            "retry_count": retry + 1,
        }

    graph = StateGraph(RAGState)

    graph.add_node("route_query", make_route_query_node(router))
    graph.add_node("retrieve_docs", retrieve_docs_with_retry_tracking)
    graph.add_node("synthesize", make_synthesize_node(synthesizer))
    graph.add_node("validate", make_validate_node(validator))

    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "retrieve_docs")
    graph.add_edge("retrieve_docs", "synthesize")
    graph.add_edge("synthesize", "validate")

    graph.add_conditional_edges(
        "validate",
        should_retry,
        {END: END, "retrieve_docs": "retrieve_docs"},
    )

    return graph.compile()


def run_langgraph_flow(query: str, vector_store: VectorStore) -> Dict[str, Any]:
    app = build_langgraph_flow(vector_store)

    initial_state: RAGState = {
        "query": query,
        "original_query": query,
        "route": "",
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "validation": {},
        "is_valid": False,
        "retry_count": 0,
    }

    logger.info("=== LangGraph Flow START | query: %s ===", query[:80])
    final_state = app.invoke(initial_state)
    logger.info("=== LangGraph Flow END | valid=%s ===", final_state["is_valid"])
    return final_state
