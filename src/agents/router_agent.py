import logging
from typing import Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm_factory import create_llm
from src.rate_limiter import llm_call_with_retry

logger = logging.getLogger(__name__)

RouteType = Literal["case_analysis", "legal_argument", "factual_lookup", "comparative"]

_VALID_ROUTES = {
    "case_analysis",
    "legal_argument",
    "factual_lookup",
    "comparative",
}

_ROUTER_PROMPT = """\
You are a legal-query router for a securities-law research system.

Classify the user's query into exactly ONE of the following categories:

  case_analysis  — asks about a specific case (parties, holding, procedural history, outcome)
  legal_argument — asks about legal reasoning, precedent, doctrine, or standards of law
  factual_lookup — asks for a specific fact: a date, name, statute number, or short phrase
  comparative    — asks to compare or contrast multiple cases, arguments, or doctrines

Reply with ONLY the category name, nothing else.

Query: {query}
"""


class RouterAgent:
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None) -> None:
        self._llm = llm or create_llm(temperature=0.0, max_output_tokens=16)

    @llm_call_with_retry
    def _call_llm(self, prompt: str) -> str:
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().lower()

    def route(self, query: str) -> RouteType:
        prompt = _ROUTER_PROMPT.format(query=query)
        raw = self._call_llm(prompt)

        label = raw.strip(" .\n").replace(" ", "_")

        if label not in _VALID_ROUTES:
            logger.warning(
                "Router returned unknown label '%s' — defaulting to 'legal_argument'.", label
            )
            return "legal_argument"

        logger.info("Route: '%s' → %s", query[:60], label)
        return label  # type: ignore[return-value]
