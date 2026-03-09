import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)


def _build_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStore:
    """
    Load-only vector store for the RAG application.
    Assumes the collection has already been populated by `ingest.py`.
    """

    def __init__(self) -> None:
        self._embeddings = _build_embeddings()
        self._store = self._load_existing()

    def _load_existing(self) -> Chroma:
        store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        count = store._collection.count()
        if count == 0:
            raise RuntimeError(
                f"Collection '{settings.collection_name}' is empty or does not exist. "
                "Run `python ingest.py` first to index your documents."
            )
        logger.info(
            "Loaded collection '%s' with %d chunks.",
            settings.collection_name,
            count,
        )
        return store

    @classmethod
    def build_from_documents(cls) -> "VectorStore":
        """
        Ingestion path — called only by ingest.py.
        Loads PDFs, chunks, embeds, and persists to ChromaDB.
        """
        from src.ingestion import load_and_chunk_pdfs  # imported here to keep app path clean

        instance = cls.__new__(cls)
        instance._embeddings = _build_embeddings()

        documents = load_and_chunk_pdfs()
        logger.info(
            "Embedding %d chunks into collection '%s' …",
            len(documents),
            settings.collection_name,
        )
        instance._store = Chroma.from_documents(
            documents=documents,
            embedding=instance._embeddings,
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info(
            "Vector store built and persisted to %s.",
            settings.chroma_persist_dir,
        )
        return instance

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        return self._store.similarity_search(query, k=k or settings.retrieval_k)

    def as_retriever(self, k: Optional[int] = None):
        return self._store.as_retriever(
            search_kwargs={"k": k or settings.retrieval_k}
        )
