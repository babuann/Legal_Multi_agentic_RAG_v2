import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage

from src.llm_factory import create_llm
from src.agents import (
    RetrieverAgent,
    RouterAgent,
    SynthesizerAgent,
    ValidatorAgent,
    ValidationResult,
)
from src.config import settings
from src.rate_limiter import llm_call_with_retry
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ResearchNote:
    sub_question: str
    route: str
    answer: str
    sources: List[str]
    confidence: float


@dataclass
class DeepAgentState:
    original_query: str
    sub_questions: List[str] = field(default_factory=list)
    research_notes: List[ResearchNote] = field(default_factory=list)
    iterations_completed: int = 0
    final_answer: str = ""
    validation: Optional[ValidationResult] = None

    @property
    def overall_confidence(self) -> float:
        if not self.research_notes:
            return 0.0
        return sum(n.confidence for n in self.research_notes) / len(self.research_notes)


_DECOMPOSE_PROMPT = """\
You are a legal research planner.

Break the following legal question into 2–4 focused sub-questions that together
cover all aspects needed to give a complete answer.
Each sub-question should be self-contained and searchable independently.

Original question: {query}

Reply with ONLY a numbered list, one sub-question per line, no preamble:
1. …
2. …
"""

_REFLECTION_PROMPT = """\
You are a legal research quality reviewer.

Original question: {original_query}

Research notes collected so far:
{notes_summary}

Identify 1–3 specific gaps or unanswered aspects in the research above.
For each gap, write ONE focused follow-up question that would address it.

Reply with ONLY a numbered list of follow-up questions, no preamble:
1. …
"""

_FINAL_SYNTHESIS_PROMPT = """\
You are a senior legal analyst.

The following research notes were gathered to answer this question:

Original question: {original_query}

Research notes:
{notes_summary}

Synthesize these notes into one cohesive, well-structured answer.
Cite note numbers inline (e.g. [Note 1]).
If notes conflict, flag the discrepancy.
Do NOT introduce information not found in the notes.
"""


class DeepAgentsOrchestrator:
    def __init__(self, vector_store: VectorStore) -> None:
        self._router = RouterAgent()
        self._retriever = RetrieverAgent(vector_store)
        self._synthesizer = SynthesizerAgent()
        self._validator = ValidatorAgent()
        self._planner_llm = create_llm(temperature=0.2, max_output_tokens=512)

    @llm_call_with_retry
    def _call_planner(self, prompt: str) -> str:
        response = self._planner_llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _decompose_query(self, query: str) -> List[str]:
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        raw = self._call_planner(prompt)

        sub_questions: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = line.lstrip("0123456789. ").strip()
            if cleaned:
                sub_questions.append(cleaned)

        if not sub_questions:
            logger.warning("Decomposition returned no sub-questions; using original query.")
            sub_questions = [query]

        logger.info("[DeepAgents] Decomposed into %d sub-questions.", len(sub_questions))
        return sub_questions

    @staticmethod
    def _heuristic_confidence(answer: str, num_chunks: int) -> float:
        if any(phrase in answer.lower() for phrase in ("not found", "insufficient", "no information")):
            return 0.3

        length_score = min(len(answer) / 800, 1.0)
        coverage_score = min(num_chunks / settings.retrieval_k, 1.0)
        return round((length_score * 0.6 + coverage_score * 0.4), 3)

    def _research_sub_question(self, sub_question: str) -> ResearchNote:
        route = self._router.route(sub_question)
        docs = self._retriever.retrieve(sub_question, route)
        answer = self._synthesizer.synthesize(sub_question, docs, route)
        sources = list({d.metadata.get("source", "unknown") for d in docs})
        confidence = self._heuristic_confidence(answer, len(docs))

        logger.info(
            "[DeepAgents] Sub-Q: '%s' | route=%s | conf=%.2f",
            sub_question[:60],
            route,
            confidence,
        )
        return ResearchNote(
            sub_question=sub_question,
            route=route,
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    @staticmethod
    def _build_notes_summary(notes: List[ResearchNote]) -> str:
        parts: List[str] = []
        for i, note in enumerate(notes, start=1):
            parts.append(
                f"[Note {i}] Sub-question: {note.sub_question}\n"
                f"Sources: {', '.join(note.sources)}\n"
                f"Answer:\n{note.answer}"
            )
        return "\n\n---\n\n".join(parts)

    def _reflect_on_gaps(self, state: DeepAgentState) -> List[str]:
        notes_summary = self._build_notes_summary(state.research_notes)
        prompt = _REFLECTION_PROMPT.format(
            original_query=state.original_query,
            notes_summary=notes_summary,
        )
        raw = self._call_planner(prompt)

        follow_ups: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            cleaned = line.lstrip("0123456789. ").strip()
            if cleaned:
                follow_ups.append(cleaned)

        logger.info("[DeepAgents] Reflection → %d follow-up questions.", len(follow_ups))
        return follow_ups

    def _final_synthesis(self, state: DeepAgentState) -> str:
        notes_summary = self._build_notes_summary(state.research_notes)
        prompt = _FINAL_SYNTHESIS_PROMPT.format(
            original_query=state.original_query,
            notes_summary=notes_summary,
        )

        @llm_call_with_retry
        def _call() -> str:
            response = self._planner_llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        return _call()

    def run(self, query: str) -> DeepAgentState:
        state = DeepAgentState(original_query=query)
        logger.info("=== DeepAgents Flow START | query: %s ===", query[:80])

        state.sub_questions = self._decompose_query(query)

        for sub_q in state.sub_questions:
            note = self._research_sub_question(sub_q)
            state.research_notes.append(note)

        state.iterations_completed = 1
        logger.info(
            "[DeepAgents] Iteration 1 complete | confidence=%.2f",
            state.overall_confidence,
        )

        for iteration in range(2, settings.max_iterations + 1):
            if state.overall_confidence >= settings.confidence_threshold:
                logger.info(
                    "[DeepAgents] Early exit at iteration %d (confidence=%.2f ≥ %.2f).",
                    iteration - 1,
                    state.overall_confidence,
                    settings.confidence_threshold,
                )
                break

            follow_up_questions = self._reflect_on_gaps(state)
            if not follow_up_questions:
                logger.info("[DeepAgents] No follow-up questions generated — stopping.")
                break

            state.sub_questions = follow_up_questions

            for sub_q in follow_up_questions:
                note = self._research_sub_question(sub_q)
                state.research_notes.append(note)

            state.iterations_completed = iteration
            logger.info(
                "[DeepAgents] Iteration %d complete | confidence=%.2f | total notes=%d",
                iteration,
                state.overall_confidence,
                len(state.research_notes),
            )

        logger.info("[DeepAgents] Synthesizing final answer from %d notes …", len(state.research_notes))
        state.final_answer = self._final_synthesis(state)

        state.validation = self._validator.validate(
            query=state.original_query,
            answer=state.final_answer,
            docs=[],
        )

        logger.info(
            "=== DeepAgents Flow END | iterations=%d | valid=%s | score=%.2f ===",
            state.iterations_completed,
            state.validation.is_valid,
            state.validation.overall_score,
        )
        return state


def run_deep_agents_flow(query: str, vector_store: VectorStore) -> DeepAgentState:
    orchestrator = DeepAgentsOrchestrator(vector_store)
    return orchestrator.run(query)
