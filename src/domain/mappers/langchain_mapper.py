from langchain_core.documents import Document

from src.domain.entities.chunk import Chunk


class LangchainMapper:
    def map(
        document: Document,
        content_embedding: list[float],
        chapter_number: int,
        text_initial_page: int,
    ) -> Chunk:
        return Chunk(
            content=document.page_content,
            embedding=content_embedding,
            page_number=int(document.metadata["page_label"]) - text_initial_page + 1,
            chapter_number=chapter_number,
        )
