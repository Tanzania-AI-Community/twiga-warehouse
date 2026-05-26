from pydantic import BaseModel


class TextChunk(BaseModel):
    content: str
    page_number: int
    chapter_number: int


class EmbeddedChunk(BaseModel):
    content: str
    page_number: int
    chapter_number: int
    embedding: list[float]
