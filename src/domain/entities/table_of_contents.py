from pydantic import BaseModel, Field


class Chapter(BaseModel):
    name: str = Field(description="Chapter name")
    number: int = Field(description="Chapter number")
    start_page: int = Field(description="Chapter start page number")


class TableOfContents(BaseModel):
    chapters: list[Chapter] = Field(description="List of chapters in the table of contents")
