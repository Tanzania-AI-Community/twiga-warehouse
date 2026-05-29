from dataclasses import dataclass
from pathlib import Path

import fitz
from tqdm import tqdm

from src.models.document import ParsedDocument, ParsedPage
from src.models.toc import Chapter, TableOfContents
from src.providers.nuextract import NuExtractClient


@dataclass(frozen=True)
class ChapterRange:
    chapter: Chapter | None
    pdf_start_page: int
    pdf_end_page: int


class HostedParser:
    PAGE_MARKDOWN_TEMPLATE = {
        "pages": [
            {
                "page_number": "integer",
                "markdown": "string",
            }
        ]
    }

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str = NuExtractClient.MODEL_NAME,
        dpi: int = 100,
        temperature: float = 0.2,
    ):
        self.nuextract_client = NuExtractClient(
            base_url=base_url,
            model_name=model_name,
            dpi=dpi,
            temperature=temperature,
        )
        self.base_url = self.nuextract_client.base_url
        self.temperature = temperature

    def parse(
        self,
        pdf_path: Path,
        table_of_contents: TableOfContents | None = None,
        first_page_number: int = 1,
    ) -> ParsedDocument:
        normalized_path = Path(pdf_path)
        markdown_output_path = self._get_markdown_output_path(pdf_path=normalized_path)
        self._initialize_markdown_output(markdown_output_path=markdown_output_path)
        pages: list[ParsedPage] = []

        with fitz.open(normalized_path) as document:
            chapter_ranges = self._build_chapter_ranges(
                table_of_contents=table_of_contents,
                total_pages=document.page_count,
                first_page_number=first_page_number,
            )

            for chapter_range in tqdm(chapter_ranges):
                page_data_urls = self._render_pages_to_data_urls(
                    document=document,
                    pdf_start_page=chapter_range.pdf_start_page,
                    pdf_end_page=chapter_range.pdf_end_page,
                )
                response_payload = self._request_chapter_parse(
                    page_data_urls=page_data_urls,
                    pdf_start_page=chapter_range.pdf_start_page,
                    pdf_end_page=chapter_range.pdf_end_page,
                    chapter=chapter_range.chapter,
                )
                chapter_pages = self._response_to_parsed_pages(
                    response_payload=response_payload,
                    pdf_start_page=chapter_range.pdf_start_page,
                    pdf_end_page=chapter_range.pdf_end_page,
                )
                self._append_chapter_to_markdown(
                    markdown_output_path=markdown_output_path,
                    chapter_pages=chapter_pages,
                )
                pages.extend(chapter_pages)

        pages.sort(key=lambda page: page.page_number)
        return ParsedDocument(pages=pages)

    @staticmethod
    def _get_markdown_output_path(
        pdf_path: Path,
    ) -> Path:
        return pdf_path.with_suffix(".md")

    @staticmethod
    def _initialize_markdown_output(
        markdown_output_path: Path,
    ) -> None:
        markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_output_path.write_text("", encoding="utf-8")

    @staticmethod
    def _append_chapter_to_markdown(
        markdown_output_path: Path,
        chapter_pages: list[ParsedPage],
    ) -> None:
        if not chapter_pages:
            return

        with markdown_output_path.open(mode="a", encoding="utf-8") as handle:
            for page in chapter_pages:
                handle.write(page.text)
            handle.write("\n")

    @staticmethod
    def _resolve_base_url(base_url: str | None) -> str:
        return NuExtractClient.resolve_base_url(base_url=base_url)

    @staticmethod
    def _build_chapter_ranges(
        table_of_contents: TableOfContents | None,
        total_pages: int,
        first_page_number: int,
    ) -> list[ChapterRange]:
        if not table_of_contents or not table_of_contents.chapters:
            return [ChapterRange(chapter=None, pdf_start_page=1, pdf_end_page=total_pages)]

        sorted_chapters = sorted(
            table_of_contents.chapters,
            key=lambda chapter: (chapter.start_page, chapter.number),
        )
        chapter_ranges: list[ChapterRange] = []

        for index, chapter in enumerate(sorted_chapters):
            pdf_start_page = max(1, chapter.start_page + first_page_number - 1)

            if index + 1 < len(sorted_chapters):
                next_chapter = sorted_chapters[index + 1]
                pdf_end_page = next_chapter.start_page + first_page_number - 2
            else:
                pdf_end_page = total_pages

            pdf_end_page = min(total_pages, pdf_end_page)
            if pdf_start_page > total_pages or pdf_start_page > pdf_end_page:
                continue

            chapter_ranges.append(
                ChapterRange(
                    chapter=chapter,
                    pdf_start_page=pdf_start_page,
                    pdf_end_page=pdf_end_page,
                )
            )

        if chapter_ranges:
            return chapter_ranges

        return [ChapterRange(chapter=None, pdf_start_page=1, pdf_end_page=total_pages)]

    @staticmethod
    def _build_prompt(
        chapter: Chapter | None,
        pdf_start_page: int,
        pdf_end_page: int,
    ) -> str:
        chapter_context = ""
        if chapter is not None:
            chapter_context = f' for chapter "{chapter.name}" (chapter {chapter.number})'

        return (
            f"Transcribe these textbook page images{chapter_context} into markdown in reading order. "
            f"The images correspond to PDF pages {pdf_start_page} through {pdf_end_page}. "
            'Return one item in "pages" per image. '
            '"page_number" must be 1-indexed within this image batch. '
            "Use markdown for text, preserve figure captions, and represent tables in readable markdown or HTML."
        )

    def _render_pages_to_data_urls(
        self,
        *,
        document: fitz.Document,
        pdf_start_page: int,
        pdf_end_page: int,
    ) -> list[str]:
        return self.nuextract_client.render_document_pages_to_data_urls(
            document=document,
            page_numbers=list(range(pdf_start_page, pdf_end_page + 1)),
        )

    def _request_chapter_parse(
        self,
        *,
        page_data_urls: list[str],
        pdf_start_page: int,
        pdf_end_page: int,
        chapter: Chapter | None,
    ) -> object:
        return self.nuextract_client.extract_structured(
            page_data_urls=page_data_urls,
            template=self.PAGE_MARKDOWN_TEMPLATE,
            instructions=self._build_prompt(
                chapter=chapter,
                pdf_start_page=pdf_start_page,
                pdf_end_page=pdf_end_page,
            ),
            temperature=self.temperature,
        )

    @staticmethod
    def _parse_response_payload(response_body: str) -> object:
        return NuExtractClient.parse_response_payload(response_body=response_body)

    @staticmethod
    def _extract_completion_text(message_content: object) -> str:
        return NuExtractClient.extract_completion_text(message_content=message_content)

    @staticmethod
    def _strip_code_fences(response_body: str) -> str:
        return NuExtractClient.strip_code_fences(response_body=response_body)

    @classmethod
    def _response_to_parsed_pages(
        cls,
        *,
        response_payload: object,
        pdf_start_page: int,
        pdf_end_page: int,
    ) -> list[ParsedPage]:
        raw_pages = cls._extract_response_pages(response_payload=response_payload)
        if not raw_pages:
            raise ValueError(
                f"Hosted OCR response did not contain any pages for PDF pages "
                f"{pdf_start_page}-{pdf_end_page}."
            )

        parsed_pages: list[ParsedPage] = []
        chapter_page_count = pdf_end_page - pdf_start_page + 1

        for default_page_number, raw_page in enumerate(raw_pages, start=1):
            page_text = cls._extract_page_text(raw_page=raw_page)
            page_number_in_chapter = cls._extract_page_number(raw_page=raw_page)

            if page_number_in_chapter is None or not 1 <= page_number_in_chapter <= chapter_page_count:
                page_number_in_chapter = default_page_number

            parsed_pages.append(
                ParsedPage(
                    page_number=pdf_start_page + page_number_in_chapter - 1,
                    text=page_text if page_text.endswith("\n") else f"{page_text}\n",
                )
            )

        return parsed_pages

    @classmethod
    def _extract_response_pages(cls, response_payload: object) -> list[object]:
        if isinstance(response_payload, list):
            return response_payload

        if not isinstance(response_payload, dict):
            return [response_payload]

        for key in ("pages", "data", "page_texts"):
            value = response_payload.get(key)
            if isinstance(value, list):
                return value

        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content")
            if isinstance(content, str):
                return cls._extract_response_pages(NuExtractClient.parse_response_payload(content))

        for key in ("markdown", "text", "content"):
            value = response_payload.get(key)
            if value is not None:
                return [value]

        return []

    @staticmethod
    def _extract_page_text(raw_page: object) -> str:
        if isinstance(raw_page, str):
            return raw_page

        if isinstance(raw_page, dict):
            for key in ("markdown", "text", "content"):
                value = raw_page.get(key)
                if isinstance(value, str):
                    return value

        return str(raw_page)

    @staticmethod
    def _extract_page_number(raw_page: object) -> int | None:
        if not isinstance(raw_page, dict):
            return None

        page_number = raw_page.get("page_number")
        if isinstance(page_number, int):
            return page_number
        if isinstance(page_number, str):
            try:
                return int(page_number)
            except ValueError:
                return None

        return None
