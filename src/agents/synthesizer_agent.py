import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.retriever_agent import RetrieverAgent
from src.llm_factory import create_llm
from src.rate_limiter import llm_call_with_retry

logger = logging.getLogger(__name__)

_BASE_INSTRUCTION = """\
You are a senior legal analyst specialising in U.S. securities law.
Answer the query using ONLY information from the provided context.
If the context does not contain enough information, say so explicitly —
do NOT fabricate citations or holdings.

CRITICAL CITATION REQUIREMENT:
Every claim, statement, or fact you present MUST be followed by a citation in the format [N], where N is the chunk number provided in the context below. If you cannot find a source for a statement, do not include it.

Query: {query}

Context:
{context}
"""

_ROUTE_SUFFIX: Dict[str, str] = {
    "case_analysis": (
        "\nStructure your answer as:\n"
        "  Parties: …\n"
        "  Issue: …\n"
        "  Holding: …\n"
        "  Key Reasoning: …\n"
        "  Outcome: …"
    ),
    "legal_argument": (
        "\nFocus on the legal standard, controlling precedents, and the"
        " argument's strengths/weaknesses. Cite chunk numbers inline."
    ),
    "factual_lookup": (
        "\nProvide a concise, direct answer (1–3 sentences). "
        "Include the source chunk number."
    ),
    "comparative": (
        "\nOrganise your answer as a structured comparison:\n"
        "  Document A: …\n"
        "  Document B: …\n"
        "  Key Similarities: …\n"
        "  Key Differences: …"
    ),
}


class SynthesizerAgent:
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None) -> None:
        self._llm = llm or create_llm()

    @llm_call_with_retry
    def _call_llm(self, prompt: str) -> str:
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def synthesize(
        self,
        query: str,
        docs: List[Document],
        route: str = "legal_argument",
    ) -> str:
        context = RetrieverAgent.format_context(docs)
        suffix = _ROUTE_SUFFIX.get(route, "")
        prompt = _BASE_INSTRUCTION.format(query=query, context=context) + suffix

        logger.info("Synthesizing answer for route='%s', %d chunks.", route, len(docs))
        answer = self._call_llm(prompt)
        logger.info("Synthesis complete (%d chars).", len(answer))
        return answer
