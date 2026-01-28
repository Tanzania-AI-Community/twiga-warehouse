from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TOGETHER_AI_API_KEY: Optional[str]
    GOOGLE_AI_API_KEY: Optional[str]
    MISTRAL_API_KEY: Optional[str]
    UNSTRUCTURED_API_KEY: Optional[str]
    UNSTRUCTURED_API_URL: Optional[str]
    INPUT_BOOKS_PATH: str
    OUTPUT_BOOKS_PATH: str

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        env_file_encoding = "utf-8"


settings = Settings()
