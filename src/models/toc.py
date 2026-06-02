from enum import Enum

from pydantic import BaseModel, Field, SecretStr


class SubChapter(BaseModel):
    name: str = Field(description="Subchapter name")
    start_page: int = Field(description="Subchapter start page number")


class Chapter(BaseModel):
    name: str = Field(description="Chapter name")
    number: int = Field(description="Chapter number")
    start_page: int = Field(description="Chapter start page number")
    subchapters: list[SubChapter] = Field(default_factory=list, description="List of subchapters")


class TableOfContents(BaseModel):
    chapters: list[Chapter] = Field(description="List of chapters in the table of contents")


class TableOfContentsParserType(str, Enum):
    GEMINI = "gemini"
    TOGETHER = "together"
    OLLAMA = "ollama"
    HOSTED = "hosted"
    NONE = "none"


class TableOfContentsParserConfig(BaseModel):
    parser_type: TableOfContentsParserType = TableOfContentsParserType.GEMINI
    api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model_name: str | None = "llama3.2"
