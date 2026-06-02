from enum import Enum
from pydantic import BaseModel


class PageContentType(str, Enum):
    PLAIN_TEXT = "text"
    MARKDOWN = "markdown"


class ParsedPage(BaseModel):
    page_number: int
    content: str | None = None
    content_type: PageContentType = PageContentType.PLAIN_TEXT


class ParsedDocument(BaseModel):
    pages: list[ParsedPage]
