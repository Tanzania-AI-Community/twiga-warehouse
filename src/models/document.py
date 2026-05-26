from pydantic import BaseModel


class ParsedPage(BaseModel):
    page_number: int
    text: str


class ParsedDocument(BaseModel):
    pages: list[ParsedPage]
