from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_CLOUD_LOCATION: str = "global"
    EMBEDDING_DIMENSIONS: int = 1024
    TOGETHER_AI_API_KEY: Optional[str] = None
    GOOGLE_AI_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    UNSTRUCTURED_API_KEY: Optional[str] = None
    UNSTRUCTURED_API_URL: Optional[str] = None
    INPUT_BOOKS_PATH: str
    OUTPUT_BOOKS_PATH: str

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        extra = "ignore"
        env_file_encoding = "utf-8"


settings = Settings()
