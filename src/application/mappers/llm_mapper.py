from langchain_core.documents import Document

from src.domain.entities.chunk import Chunk


class LLMMapper:
    def map(
        page_number: int,
        chapter_number: int,
        text_initial_page: int,
        content_text: str,
        content_embedding: list[float],
    ) -> Chunk:
        return Chunk(
            content=content_text,
            embedding=content_embedding,
            page_number=page_number + text_initial_page - 1,
            chapter_number=chapter_number,
        )
