import base64
import json
from dataclasses import dataclass
from pathlib import Path

import fitz
from openai import OpenAI

from src.config.settings import settings
from src.models.document import ParsedDocument, ParsedPage
from src.models.toc import Chapter, TableOfContents


@dataclass(frozen=True)
class ChapterRange:
    chapter: Chapter | None
    pdf_start_page: int
    pdf_end_page: int


class HostedParser:
    MODEL_NAME = "numind/NuExtract3"

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str = MODEL_NAME,
        dpi: int = 100,
        temperature: float = 0.2,
    ):
        resolved_base_url = self._resolve_base_url(base_url=base_url)
        if not resolved_base_url:
            raise ValueError("CUSTOM_OCR_MODEL_URL is required to use the hosted OCR parser.")

        self.client = OpenAI(
            api_key=settings.CUSTOM_OCR_MODEL_API_KEY,
            base_url=resolved_base_url,
        )
        self.base_url = resolved_base_url
        self.model_name = model_name
        self.dpi = dpi
        self.temperature = temperature

    def parse(
        self,
        pdf_path: Path,
        table_of_contents: TableOfContents | None = None,
        first_page_number: int = 1,
    ) -> ParsedDocument:
        normalized_path = Path(pdf_path)
        pages: list[ParsedPage] = []

        with fitz.open(normalized_path) as document:
            chapter_ranges = self._build_chapter_ranges(
                table_of_contents=table_of_contents,
                total_pages=document.page_count,
                first_page_number=first_page_number,
            )

            for chapter_range in chapter_ranges:
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
                pages.extend(
                    self._response_to_parsed_pages(
                        response_payload=response_payload,
                        pdf_start_page=chapter_range.pdf_start_page,
                        pdf_end_page=chapter_range.pdf_end_page,
                    )
                )

        pages.sort(key=lambda page: page.page_number)
        return ParsedDocument(pages=pages)

    @staticmethod
    def _resolve_base_url(base_url: str | None) -> str:
        resolved_base_url = (base_url or settings.CUSTOM_OCR_MODEL_URL or "").strip().rstrip("/")
        if not resolved_base_url:
            return ""
        if resolved_base_url.endswith("/v1"):
            return resolved_base_url
        return f"{resolved_base_url}/v1"

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
            'Return valid JSON with a top-level "pages" array. '
            'Each item in "pages" must have "page_number" (1-indexed within this image batch) and '
            '"markdown". Do not add code fences or commentary.'
        )

    def _render_pages_to_data_urls(
        self,
        *,
        document: fitz.Document,
        pdf_start_page: int,
        pdf_end_page: int,
    ) -> list[str]:
        data_urls: list[str] = []

        for page_index in range(pdf_start_page - 1, pdf_end_page):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
            png_base64 = base64.b64encode(pixmap.tobytes("png")).decode("utf-8")
            data_urls.append(f"data:image/png;base64,{png_base64}")

        return data_urls

    def _request_chapter_parse(
        self,
        *,
        page_data_urls: list[str],
        pdf_start_page: int,
        pdf_end_page: int,
        chapter: Chapter | None,
    ) -> object:
        if not page_data_urls:
            raise ValueError("Hosted OCR parser received an empty chapter page range.")

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_prompt(
                                chapter=chapter,
                                pdf_start_page=pdf_start_page,
                                pdf_end_page=pdf_end_page,
                            ),
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            }
                            for data_url in page_data_urls
                        ],
                    ],
                }
            ],
        )
        response_body = self._extract_completion_text(response.choices[0].message.content)
        if not response_body:
            raise ValueError("Hosted OCR response did not contain any text.")

        return self._parse_response_payload(response_body=response_body)

    @staticmethod
    def _parse_response_payload(response_body: str) -> object:
        candidates = [response_body.strip()]
        fenced_candidate = HostedParser._strip_code_fences(response_body=response_body)
        if fenced_candidate != candidates[0]:
            candidates.append(fenced_candidate)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return {"text": response_body}

    @staticmethod
    def _extract_completion_text(message_content: object) -> str:
        if isinstance(message_content, str):
            return message_content

        if isinstance(message_content, list):
            text_parts: list[str] = []

            for item in message_content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

            return "\n".join(text_parts)

        return ""

    @staticmethod
    def _strip_code_fences(response_body: str) -> str:
        stripped = response_body.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            return stripped

        return "\n".join(lines[1:-1]).strip()

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
                try:
                    return cls._extract_response_pages(json.loads(content))
                except json.JSONDecodeError:
                    return [content]

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

        return None
