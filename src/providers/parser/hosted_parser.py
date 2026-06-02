import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import fitz
from pydantic import ValidationError
from tqdm import tqdm

from src.models.document import PageContentType, ParsedDocument, ParsedPage
from src.models.toc import Chapter, TableOfContents
from src.providers.nuextract import NuExtractClient
from src.providers.llm.together import client as TogetherClient

JSON_CHAPTER_SAVE_NAME = "chapter_{chapter_number}.json"
MAX_CONCURRENT_PAGE_TASKS = 5


@dataclass(frozen=True)
class ChapterRange:
    chapter: Chapter | None
    pdf_start_page: int
    pdf_end_page: int


@dataclass(frozen=True)
class PageParseRequest:
    pdf_page_number: int
    chapter_page_number: int
    page_data_url: str


class HostedParser:
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str = NuExtractClient.MODEL_NAME,
        dpi: int = 100,
        temperature: float = 0.0,
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
        checkpoints_path: Path | None = None,
    ) -> ParsedDocument:
        normalized_path = Path(pdf_path)
        pages: list[ParsedPage] = []

        with fitz.open(normalized_path) as document:
            chapter_ranges = self._build_chapter_ranges(
                table_of_contents=table_of_contents,
                total_pages=document.page_count,
                first_page_number=first_page_number,
            )
            for chapter_range in tqdm(chapter_ranges):
                chapter_number = self._get_chapter_number(chapter=chapter_range.chapter)
                json_chapter_path = self._build_checkpoint_path(
                    checkpoints_path=checkpoints_path,
                    chapter_number=chapter_number,
                )
                parsed_chapter = self._load_chapter_from_checkpoint(json_chapter_path=json_chapter_path)

                if parsed_chapter:
                    pages.extend(parsed_chapter.pages)
                    continue

                parsed_chapter = asyncio.run(
                    self._parse_chapter_async(
                        document=document,
                        chapter_range=chapter_range,
                    )
                )
                self._save_chapter_to_checkpoint(
                    json_chapter_path=json_chapter_path,
                    chapter_document=parsed_chapter,
                    chapter_number=chapter_number,
                )
                pages.extend(parsed_chapter.pages)

        return ParsedDocument(pages=pages)

    @staticmethod
    def _build_checkpoint_path(
        checkpoints_path: Path | None,
        chapter_number: int | None,
    ) -> Path | None:
        if checkpoints_path is None or chapter_number is None:
            return None

        return checkpoints_path / JSON_CHAPTER_SAVE_NAME.format(chapter_number=chapter_number)

    @staticmethod
    def _load_chapter_from_checkpoint(json_chapter_path: Path | None) -> ParsedDocument | None:
        if json_chapter_path is None or not json_chapter_path.exists():
            return None

        try:
            with json_chapter_path.open(mode="r", encoding="utf-8") as json_file:
                parsed_document = json.load(json_file)
        except (json.JSONDecodeError, OSError):
            return None

        try:
            return ParsedDocument.model_validate(parsed_document)
        except ValidationError:
            return None

    @staticmethod
    def _save_chapter_to_checkpoint(
        json_chapter_path: Path | None,
        chapter_document: ParsedDocument,
        chapter_number: int | None = None,
    ) -> None:
        if not chapter_document or chapter_number is None or json_chapter_path is None:
            return

        with json_chapter_path.open(mode="w", encoding="utf-8") as json_file:
            json.dump(chapter_document.model_dump(), json_file, ensure_ascii=False, indent=4)

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
    def _get_chapter_number(chapter: Chapter | None) -> int | None:
        if chapter is None:
            return None

        return chapter.number

    @staticmethod
    def _build_ocr_prompt(
        chapter: Chapter | None,
        pdf_page_number: int,
        chapter_page_number: int,
    ) -> str:
        chapter_context = ""
        if chapter is not None:
            chapter_context = f' for chapter "{chapter.name}" (chapter {chapter.number})'

        prompt = (
            f"Transcribe this single textbook page image{chapter_context} into markdown in reading order. "
            f"The image corresponds to PDF page {pdf_page_number}, which is page {chapter_page_number} within the chapter. "
            "Return ONLY the markdown text for this page. No JSON. No markdown code fences. No commentary before or after the markdown.\n\n"
            "Rules:\n"
            "- Preserve figure captions.\n"
            "- Represent tables in readable markdown or HTML.\n"
            "- Prefer single quotes for HTML attributes.\n"
            "- For image tags, write <img src='1.png' alt='description'/>.\n"
        )
        return prompt

    def _render_page_to_data_url(
        self,
        *,
        document: fitz.Document,
        pdf_page_number: int,
    ) -> str:
        return self.nuextract_client.render_document_pages_to_data_urls(
            document=document,
            page_numbers=[pdf_page_number],
        )[0]

    def _request_page_ocr(
        self,
        *,
        page_data_url: str,
        pdf_page_number: int,
        chapter_page_number: int,
        chapter: Chapter | None,
    ) -> str:
        response = self.nuextract_client.client.chat.completions.create(
            model=self.nuextract_client.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_ocr_prompt(
                                chapter=chapter,
                                pdf_page_number=pdf_page_number,
                                chapter_page_number=chapter_page_number,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": page_data_url},
                        },
                    ],
                }
            ],
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
            timeout=600,
        )
        response_body = self.nuextract_client.extract_completion_text(response.choices[0].message.content)
        if not response_body:
            raise ValueError(f"OCR response was empty for PDF page {pdf_page_number}.")

        markdown = NuExtractClient.strip_code_fences(response_body=response_body).strip()
        if not markdown:
            raise ValueError(f"OCR response was blank after cleanup for PDF page {pdf_page_number}.")

        return markdown

    async def _parse_chapter_async(
        self,
        *,
        document: fitz.Document,
        chapter_range: ChapterRange,
    ) -> ParsedDocument:
        page_requests = self._build_page_parse_requests(
            document=document,
            chapter_range=chapter_range,
        )
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGE_TASKS)
        pages = await asyncio.gather(
            *[
                self._parse_page_async(
                    page_request=page_request,
                    chapter=chapter_range.chapter,
                    semaphore=semaphore,
                )
                for page_request in page_requests
            ]
        )
        return ParsedDocument(pages=pages)

    def _build_page_parse_requests(
        self,
        *,
        document: fitz.Document,
        chapter_range: ChapterRange,
    ) -> list[PageParseRequest]:
        page_requests: list[PageParseRequest] = []

        for chapter_page_number, pdf_page_number in enumerate(
            range(chapter_range.pdf_start_page, chapter_range.pdf_end_page + 1),
            start=1,
        ):
            page_requests.append(
                PageParseRequest(
                    pdf_page_number=pdf_page_number,
                    chapter_page_number=chapter_page_number,
                    page_data_url=self._render_page_to_data_url(
                        document=document,
                        pdf_page_number=pdf_page_number,
                    ),
                )
            )

        return page_requests

    async def _parse_page_async(
        self,
        *,
        page_request: PageParseRequest,
        chapter: Chapter | None,
        semaphore: asyncio.Semaphore,
    ) -> ParsedPage:
        async with semaphore:
            raw_markdown = await asyncio.to_thread(
                self._request_page_ocr,
                page_data_url=page_request.page_data_url,
                pdf_page_number=page_request.pdf_page_number,
                chapter_page_number=page_request.chapter_page_number,
                chapter=chapter,
            )
            cleaned_markdown = await asyncio.to_thread(
                self._clean_page_with_retry,
                raw_markdown=raw_markdown,
                chapter=chapter,
                pdf_page_number=page_request.pdf_page_number,
            )

        return ParsedPage(
            page_number=page_request.chapter_page_number,
            content=cleaned_markdown,
            content_type=PageContentType.MARKDOWN,
        )

    @staticmethod
    def _clean_page_with_retry(
        raw_markdown: str,
        chapter: Chapter | None,
        pdf_page_number: int,
    ) -> str:
        cleaned_markdown = TogetherClient.clean_page(
            markdown=raw_markdown,
            chapter=chapter,
        )
        if cleaned_markdown is not None:
            return cleaned_markdown

        cleaned_markdown = TogetherClient.clean_page(
            markdown=raw_markdown,
            chapter=chapter,
        )
        if cleaned_markdown is not None:
            return cleaned_markdown

        raise ValueError(f"LLM cleaner returned null twice for PDF page {pdf_page_number}.")
