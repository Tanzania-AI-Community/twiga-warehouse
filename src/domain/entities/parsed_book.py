from pydantic import BaseModel

from src.domain.entities.chunk import Chunk


class ParsedBook(BaseModel):
    title: str
    author: str
    chunks: list[Chunk]
