"""Centralized configuration management using Pydantic settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment_name: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # OpenAI Fallback
    openai_api_key: str = ""

    # Azure AI Search
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index_name: str = "logistics-knowledge-base"

    # Application
    log_level: str = "INFO"
    environment: str = "development"
    chunk_size: int = 512
    chunk_overlap: int = 50
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    top_k_results: int = 5

    @property
    def use_azure_openai(self) -> bool:
        """Check if Azure OpenAI credentials are configured."""
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    @property
    def use_azure_search(self) -> bool:
        """Check if Azure AI Search credentials are configured."""
        return bool(self.azure_search_endpoint and self.azure_search_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
