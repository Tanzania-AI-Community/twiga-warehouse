from pydantic import BaseModel


class Chunk(BaseModel):
    content: str
    embedding: list[float]
    page_number: int
    chapter_number: int
