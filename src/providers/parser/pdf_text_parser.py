from pathlib import Path

from pypdf import PdfReader

from src.models.document import ParsedDocument, ParsedPage
from src.models.toc import TableOfContents


class PdfTextParser:
    def parse(
        self,
        pdf_path: Path,
        table_of_contents: TableOfContents | None = None,
        first_page_number: int = 1,
    ) -> ParsedDocument:
        del table_of_contents, first_page_number
        reader = PdfReader(stream=pdf_path)
        pages: list[ParsedPage] = []

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(ParsedPage(page_number=page_index, text=page_text))

        return ParsedDocument(pages=pages)
