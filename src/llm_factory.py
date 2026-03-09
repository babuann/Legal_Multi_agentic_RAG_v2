import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

logger = logging.getLogger(__name__)


def create_llm(
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    resolved_temp = temperature if temperature is not None else settings.llm_temperature
    resolved_tokens = max_output_tokens or settings.llm_max_output_tokens

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=resolved_temp,
        google_api_key=settings.google_api_key,
        max_output_tokens=resolved_tokens,
    )
