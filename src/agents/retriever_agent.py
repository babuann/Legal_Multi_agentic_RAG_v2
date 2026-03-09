import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.config import settings
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        route: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        base_k = k or settings.retrieval_k

        effective_k = {
            "comparative": base_k * 2,
            "case_analysis": base_k + 2,
        }.get(route, base_k)

        logger.info(
            "Retrieving k=%d chunks for route='%s', query='%s'",
            effective_k,
            route,
            query[:70],
        )

        docs = self._store.similarity_search(query, k=effective_k)

        sources = {d.metadata.get("source", "unknown") for d in docs}
        logger.info("Retrieved %d chunks from: %s", len(docs), sources)

        return docs

    @staticmethod
    def format_context(docs: List[Document]) -> str:
        parts: List[str] = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(
                f"[{i}] Source: {source} | Page: {page}\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)
