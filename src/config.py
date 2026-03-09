from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str

    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 2048

    max_requests_per_minute: int = 14

    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # original base model
    embedding_model: str = "./models/legal-contrastive"  # fine-tuned on legal documents

    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "legal_documents_v2"  # to trigger rebuild with new embeddings

    chunk_size: int = 800
    chunk_overlap: int = 150

    data_dir: str = "./data"

    retrieval_k: int = 5

    max_iterations: int = 1
    confidence_threshold: float = 0.80


settings = Settings()
