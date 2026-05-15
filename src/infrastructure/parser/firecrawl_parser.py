import io
import logging
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from langchain_core.documents import Document
from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions 
from src.config.settings import settings

logging.basicConfig(level=logging.INFO)

class FirecrawlParser:
    def __init__(self, book_path: Path):
        self.client = Firecrawl(api_key=settings.FIRECRAWL_API_KEY)
        self.book_path = Path(book_path)

    def load(self) -> list[Document]:
        reader = PdfReader(self.book_path)
        documents = []

        for i, page in enumerate(reader.pages):
            page_num = i + 1

            writer = PdfWriter()
            writer.add_page(page)
            
            buffer = io.BytesIO()
            writer.write(buffer)
            buffer.seek(0)
            file_bytes = buffer.read()

            try:

                scrape_options = ScrapeOptions(
                    formats=["markdown"],
                    parsers=[{"type": "pdf", "mode": "ocr"}]
                )

                parsed_doc = self.client.parse(
                    file_bytes,
                    filename=f"page_{page_num}.pdf",
                    content_type="application/pdf",
                    options=scrape_options
                )

                markdown_text = getattr(parsed_doc, 'markdown', '')
                if not markdown_text and isinstance(parsed_doc, dict):
                    markdown_text = parsed_doc.get("markdown", "")

                documents.append(
                    Document(
                        page_content=markdown_text,
                        metadata={
                            "page_number": page_num,
                            "page_label": page_num,
                            "source": self.book_path.name
                        }
                    )
                )
            except Exception as e:
                logging.error(f"Failed to parse page {page_num} with Firecrawl: {e}")
                continue

        return documents