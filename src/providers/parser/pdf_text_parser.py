from pathlib import Path

from pypdf import PdfReader

from src.models.document import PageContentType, ParsedDocument, ParsedPage
from src.models.toc import TableOfContents


class PdfTextParser:
    def parse(
        self,
        pdf_path: Path,
        table_of_contents: TableOfContents | None = None,
        first_page_number: int = 1,
        checkpoints_path: Path | None = None,
    ) -> ParsedDocument:
        del table_of_contents, first_page_number, checkpoints_path
        reader = PdfReader(stream=pdf_path)
        pages: list[ParsedPage] = []

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(ParsedPage(page_number=page_index, content=page_text, content_type=PageContentType.PLAIN_TEXT))

        return ParsedDocument(pages=pages)
