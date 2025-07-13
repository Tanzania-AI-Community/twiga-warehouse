from abc import abstractmethod
from enum import Enum

from pydantic import BaseModel

from src.domain.entities.chunk import Chunk
from src.domain.entities.table_of_contents import TableOfContents


class ChunkerType(str, Enum):
    UNSTRUCTURED = "unstructured"
    LANGCHAIN = "langchain"


class ChunkerConfig(BaseModel):
    chunker_type: ChunkerType


class Chunker:
    def __init__(self, config: ChunkerConfig):
        self.config = config

    @abstractmethod
    def chunk(book_path: str, table_of_contents: TableOfContents = None, text_initial_page: int = None) -> list[Chunk]:
        pass


class EmptyChunkerResponse(Exception):
    def __init__(self, book_path: str, chunker_config: ChunkerConfig) -> None:
        self.book_path = book_path
        self.chunker_config = chunker_config

        msg = f"Chunker ({chunker_config.chunker_type}) produced empty chunk list for book: {book_path}."
        super().__init__(msg)
        