import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm_factory import create_llm
from src.rate_limiter import llm_call_with_retry

logger = logging.getLogger(__name__)

_VALIDATOR_PROMPT = """\
You are a strict quality-control reviewer for a legal research system.

Evaluate the answer below against the query and the retrieved context.

Query: {query}

Answer:
{answer}

CRITICAL VALIDATION RULES:
1. Relevance: Does the answer directly address the query?
2. Grounding: Is EVERY claim supported by the provided context? If a claim is not explicitly in the context, it is NOT grounded.
3. Completeness: Does it cover all key aspects of the question?

Score each criterion on a scale of 0.0 – 1.0.

Also provide:
  is_valid  — true if ALL three scores ≥ 0.7, otherwise false
  feedback  — one concise sentence explaining the main weakness (if any). If grounding is low, specify which claim is unsupported.
  refined_query — if is_valid is false, a better-rephrased query that might
                  surface missing context on the next retrieval pass.

Reply with VALID JSON only, no markdown fences:
{{
  "relevance": <float>,
  "grounding": <float>,
  "completeness": <float>,
  "is_valid": <bool>,
  "feedback": "<string>",
  "refined_query": "<string>"
}}
"""


@dataclass
class ValidationResult:
    relevance: float
    grounding: float
    completeness: float
    is_valid: bool
    feedback: str
    refined_query: str

    @property
    def overall_score(self) -> float:
        return round((self.relevance + self.grounding + self.completeness) / 3, 3)


class ValidatorAgent:
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None) -> None:
        self._llm = llm or create_llm(temperature=0.0, max_output_tokens=512)

    @llm_call_with_retry
    def _call_llm(self, prompt: str) -> str:
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _parse_response(self, raw: str, query: str) -> ValidationResult:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            return ValidationResult(
                relevance=float(data.get("relevance", 0.5)),
                grounding=float(data.get("grounding", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                is_valid=bool(data.get("is_valid", False)),
                feedback=str(data.get("feedback", "Parse error.")),
                refined_query=str(data.get("refined_query", query)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Validator JSON parse failed (%s) — using safe defaults.", exc)
            return ValidationResult(
                relevance=0.5,
                grounding=0.5,
                completeness=0.5,
                is_valid=False,
                feedback="Could not parse validator output.",
                refined_query=query,
            )

    def validate(
        self,
        query: str,
        answer: str,
        docs: List[Document],
    ) -> ValidationResult:
        prompt = _VALIDATOR_PROMPT.format(query=query, answer=answer)
        raw = self._call_llm(prompt)
        result = self._parse_response(raw, query)

        logger.info(
            "Validation → relevance=%.2f, grounding=%.2f, completeness=%.2f, valid=%s",
            result.relevance,
            result.grounding,
            result.completeness,
            result.is_valid,
        )
        return result
