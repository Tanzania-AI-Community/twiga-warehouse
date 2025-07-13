import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TOGETHER_AI_API_KEY: str
    UNSTRUCTURED_API_KEY: str
    UNSTRUCTURED_API_URL: str

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "../..", ".env")
        env_file_encoding = "utf-8"


settings = Settings()
