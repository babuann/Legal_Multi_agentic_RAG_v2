import logging
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

_LEGAL_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def load_and_chunk_pdfs(data_dir: Optional[str] = None) -> List[Document]:
    resolved_dir = Path(data_dir or settings.data_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {resolved_dir}")

    pdf_paths = sorted(resolved_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {resolved_dir}")

    logger.info("Found %d PDF(s) in %s", len(pdf_paths), resolved_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_LEGAL_SEPARATORS,
        length_function=len,
    )

    all_chunks: List[Document] = []

    for pdf_path in pdf_paths:
        logger.info("Loading: %s", pdf_path.name)
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as exc:
            logger.warning("Failed to load %s: %s", pdf_path.name, exc)
            continue

        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata.setdefault("source", pdf_path.name)

        all_chunks.extend(chunks)
        logger.info("  → %d chunks from %s", len(chunks), pdf_path.name)

    logger.info("Total chunks: %d", len(all_chunks))
    return all_chunks
