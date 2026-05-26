from abc import ABC, abstractmethod

from src.models.chunk import TextChunk
from src.models.document import ParsedDocument
from src.models.toc import TableOfContents


class Chunker(ABC):
    @abstractmethod
    def chunk(
        self,
        parsed_document: ParsedDocument,
        table_of_contents: TableOfContents,
        first_page_number: int,
        last_page_number: int | None = None,
    ) -> list[TextChunk]:
        raise NotImplementedError


def get_initial_chapter_page(
    table_of_contents: TableOfContents,
    first_page_number: int,
) -> int:
    if table_of_contents.chapters and first_page_number:
        return table_of_contents.chapters[0].start_page + first_page_number
    if first_page_number:
        return first_page_number + 1
    return 0


def get_page_chapter_number(
    page_number: int,
    first_page_number: int,
    table_of_contents: TableOfContents,
) -> int:
    chapter_number = 0

    for chapter in table_of_contents.chapters:
        if page_number < chapter.start_page + first_page_number - 1:
            break
        chapter_number = chapter.number

    return chapter_number
