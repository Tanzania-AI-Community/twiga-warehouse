from abc import abstractmethod
from enum import Enum

from pydantic import BaseModel

from src.domain.entities.chunk import Chunk


class Chunker:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def chunk(book_path: str) -> list[Chunk]:
        pass


class ChunkerType(str, Enum):
    UNSTRUCTURED = "unstructured"
    LANGCHAIN = "langchain"


class ChunkerConfig(BaseModel):
    chunker_type: ChunkerType


class EmptyChunkerResponse(Exception):
    def __init__(self, book_path: str, chunker_config: ChunkerConfig) -> None:
        self.book_path = book_path
        self.chunker_config = chunker_config

        msg = f"Chunker ({chunker_config.chunker_type}) produced empty chunk list for book: {book_path}."
        super().__init__(msg)
        