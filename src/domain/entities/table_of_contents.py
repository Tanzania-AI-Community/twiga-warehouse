from enum import Enum

from pydantic import BaseModel, Field, SecretStr


class Chapter(BaseModel):
    name: str = Field(description="Chapter name")
    number: int = Field(description="Chapter number")
    start_page: int = Field(description="Chapter start page number")


class TableOfContents(BaseModel):
    chapters: list[Chapter] = Field(description="List of chapters in the table of contents")


class TableOfContentsParserType(str, Enum):
    GEMINI = "gemini"
    TOGETHER = "together"
    OLLAMA = "ollama"
    NONE = "none"


class TableOfContentsParserConfig(BaseModel):
    parser_type: TableOfContentsParserType = TableOfContentsParserType.GEMINI
    api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model_name: str | None = "llama3.2"
