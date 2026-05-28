from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.chunk import TextChunk
from src.models.document import ParsedDocument
from src.models.toc import TableOfContents
from src.providers.chunker.base import Chunker, get_initial_chapter_page, get_page_chapter_number


class LangchainChunker(Chunker):
    UNWANTED_TEXT = [
        "FOR ONLINE USE ONLY",
        "DO NOT DUPLICATE",
        "PROPERTY OF THE UNITED REPUBLIC OF TANZANIA GOBVERNMENT",
        "Ministry of Education, Science and Technology",
        "For Online Use Only",
        "Student’s Book Form Two",
        "Geography for Secondary Schools",
    ]
    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 30, min_length_to_include: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_length_to_include = min_length_to_include
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.UNWANTED_TEXT + self.DEFAULT_SEPARATORS,
            keep_separator=False,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(
        self,
        parsed_document: ParsedDocument,
        table_of_contents: TableOfContents,
        first_page_number: int,
        last_page_number: int | None = None,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        initial_chapter_page = get_initial_chapter_page(
            table_of_contents=table_of_contents,
            first_page_number=first_page_number,
        )

        for page in parsed_document.pages:
            if page.page_number < initial_chapter_page - 1:
                continue
            if last_page_number is not None and page.page_number > last_page_number:
                break

            cleaned_text = page.text
            for unwanted_text in self.UNWANTED_TEXT:
                cleaned_text = cleaned_text.replace(unwanted_text, "")

            if len(cleaned_text) < self.min_length_to_include:
                continue

            chapter_number = get_page_chapter_number(
                page_number=page.page_number,
                first_page_number=first_page_number,
                table_of_contents=table_of_contents,
            )
            page_chunks = self.text_splitter.split_text(text=cleaned_text)

            for page_chunk in page_chunks:
                if len(page_chunk) < self.min_length_to_include:
                    continue

                chunks.append(
                    TextChunk(
                        content=page_chunk,
                        page_number=page.page_number - first_page_number + 1,
                        chapter_number=chapter_number,
                    )
                )

        return chunks
