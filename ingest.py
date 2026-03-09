"""
Ingestion script — run once before starting the application.

    python ingest.py

Loads all PDFs from the data directory, chunks them, embeds with the
fine-tuned legal model, and persists to ChromaDB.
"""
import logging
import sys

from src.config import settings
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for _noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Ingestion pipeline starting ===")
    logger.info("Embedding model : %s", settings.embedding_model)
    logger.info("Collection      : %s", settings.collection_name)
    logger.info("ChromaDB path   : %s", settings.chroma_persist_dir)
    logger.info("Data directory  : %s", settings.data_dir)

    VectorStore.build_from_documents()

    logger.info("=== Ingestion complete. You can now run the application. ===")


if __name__ == "__main__":
    main()
