from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
    )
    lancedb_dir: str
    lancedb_table: str
    embedding_model: str
    embedding_dimension: int
    azure_openai_embedding_deployment: str
    azure_openai_embedding_api_version: str
    azure_openai_chat_deployment: str
    azure_openai_chat_api_version: str
    azure_openai_endpoint: str
    azure_openai_api_key: str


@lru_cache
def get_settings() -> Settings:
    # Use lru_cache to avoid loading .env file for every request
    return Settings()
