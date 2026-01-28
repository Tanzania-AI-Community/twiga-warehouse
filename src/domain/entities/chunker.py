from abc import abstractmethod
from pathlib import Path
from enum import Enum

from pydantic import BaseModel

from src.domain.entities.chunk import Chunk
from src.domain.entities.table_of_contents import TableOfContents


class ChunkerType(str, Enum):
    UNSTRUCTURED = "unstructured"
    LANGCHAIN = "langchain"
    LLM = "llm"
    MATHEMATICAL = "mathematical"


class ChunkerConfig(BaseModel):
    chunker_type: ChunkerType
    last_page_number: int | None = None
    llm_model_name: str | None = None
    embedding_model_name: str | None = None
    page_batch_size: int | None = None


class Chunker:
    def __init__(self, config: ChunkerConfig):
        self.config = config

    @abstractmethod
    def chunk(
        self,
        book_path: Path,
        table_of_contents: TableOfContents = None,
        text_initial_page: int = None,
    ) -> list[Chunk]:
        pass


class EmptyChunkerResponse(Exception):
    def __init__(self, book_path: Path, chunker_config: ChunkerConfig) -> None:
        self.book_path = book_path
        self.chunker_config = chunker_config

        msg = f"Chunker ({chunker_config.chunker_type}) produced empty chunk list for book: {book_path}."
        super().__init__(msg)
        
