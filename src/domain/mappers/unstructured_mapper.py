from langchain_core.documents import Document

from src.domain.entities.chunk import Chunk


class UnstructuredMapper:
    def map(document: Document, content_embedding: list[float]) -> Chunk:
        return Chunk(
            content=document.page_content,
            embedding=content_embedding,
            page_number=document.metadata["page_number"],
        )

    def map(document: Document, content_embedding: list[float], chapter_number: int) -> Chunk:
        return Chunk(
            content=document.page_content,
            embedding=content_embedding,
            page_number=document.metadata["page_number"],
            chapter_number=chapter_number,
        )
