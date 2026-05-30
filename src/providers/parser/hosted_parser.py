import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from tqdm import tqdm

from src.models.document import PageContentType, ParsedDocument, ParsedPage
from src.models.toc import Chapter, TableOfContents
from src.providers.nuextract import NuExtractClient

JSON_CHAPTER_SAVE_NAME = "chapter_{chapter_number}.json"


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
                json_chapter_path = checkpoints_path / JSON_CHAPTER_SAVE_NAME.format(
                    chapter_number=chapter_range.chapter.number if chapter_range.chapter else None
                )
                chapter_pages = self._load_chapter_from_checkpoint(json_chapter_path=json_chapter_path)
                
                if not chapter_pages:
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
                    )
                    self._save_chapter_to_checkpoint(
                        json_chapter_path=json_chapter_path,
                        chapter_pages=chapter_pages,
                        chapter_number=chapter_range.chapter.number if chapter_range.chapter else None,
                    )

                pages.extend(chapter_pages)

        pages.sort(key=lambda page: page.page_number)
        return ParsedDocument(pages=pages)

    @staticmethod
    def _load_chapter_from_checkpoint(json_chapter_path: Path) -> list[ParsedPage]:
        return []
        if not json_chapter_path.exists():
            return []

        chapter_pages: list[ParsedPage] = []
        with json_chapter_path.open(mode="r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    page_data = json.loads(line)
                    chapter_pages.append(ParsedPage.model_validate(page_data))
                except json.JSONDecodeError:
                    continue

        return chapter_pages

    @staticmethod
    def _save_chapter_to_checkpoint(
        json_chapter_path: Path,
        chapter_pages: list[ParsedPage],
        chapter_number: int | None = None,
    ) -> None:
        if not chapter_pages or chapter_number is None:
            return

        with json_chapter_path.open(mode="a", encoding="utf-8") as handle:
            for page in chapter_pages:
                json.dump(page.model_dump(), handle)
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
                    pdf_end_page=pdf_start_page+1,
                )
            )
            break

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

        prompt = (
            f"Transcribe these textbook page images{chapter_context} into markdown in reading order. "
            f"The images correspond to PDF pages {pdf_start_page} through {pdf_end_page}. "
            '"page_number" must be 1-indexed within this image batch. '
            "Use markdown for text, preserve figure captions, and represent tables in readable markdown or HTML.\n\n"

            "Return ONLY a valid JSON object. No markdown code fences. No commentary before or after the JSON.\n"
            "The response must parse successfully with Python json.loads(response_text).\n\n"

            "Use exactly this JSON shape:\n"
            '{"pages":[{"page_number":1,"markdown":"..."}]}\n\n'

            "Rules:\n"
            "- Return one item in pages per image.\n"
            "- Each page object must contain exactly these keys: page_number and markdown.\n"
            "- Do not duplicate keys.\n"
            "- The markdown value must be a valid JSON string.\n"
            "- Represent newlines inside markdown as \\n, not as raw line breaks.\n"
            "- If markdown contains double quotes, escape them as \\\".\n"
            "- Prefer single quotes for HTML attributes inside markdown to avoid JSON escaping problems.\n"
            "- For example, write <figure data-type='image' data-id='1'>, not <figure data-type=\"image\" data-id=\"1\">.\n"
            "- For image tags, write <img src='1.png' alt='description'/>.\n"
        )
        return prompt

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

    @classmethod
    def _response_to_parsed_pages(
        cls,
        *,
        response_payload: object,
        # pdf_start_page: int,
        # pdf_end_page: int,
    ) -> list[ParsedPage]:
        if "text" not in response_payload:
            raise RuntimeError("Expected to have 'text' in payload.")

        loaded_payload: dict[str, Any] = response_payload["text"]

        if "pages" not in loaded_payload:
            raise RuntimeError("Expected to have 'pages' in text payload.")

        chapter_pages: list[dict, int | str] = json.loads(loaded_payload)["pages"]
        print(chapter_pages)

        cleaned_chapter_pages: list[ParsedPage] = [
            ParsedPage(
                page_number=chapter_page["page_number"],
                content=chapter_page["markdown"],
                content_type=PageContentType.MARKDOWN,
            )
            for chapter_page
            in chapter_pages
        ]
        return cleaned_chapter_pages

        # raw_pages = cls._extract_response_pages(response_payload=response_payload)
        # if not raw_pages:
        #     raise ValueError(
        #         f"Hosted OCR response did not contain any pages for PDF pages "
        #         f"{pdf_start_page}-{pdf_end_page}."
        #     )
        
        # print()
        # print(raw_pages)
        # print()

        # parsed_pages: list[ParsedPage] = []
        # chapter_page_count = pdf_end_page - pdf_start_page + 1

        # for default_page_number, raw_page in enumerate(raw_pages, start=1):
        #     print(raw_page)
        #     page_text = cls._extract_page_text(raw_page=raw_page)
        #     page_number_in_chapter = cls._extract_page_number(raw_page=raw_page)

        #     if page_number_in_chapter is None or not 1 <= page_number_in_chapter <= chapter_page_count:
        #         page_number_in_chapter = default_page_number

        #     parsed_pages.append(
        #         ParsedPage(
        #             page_number=page_number_in_chapter,
        #             content=raw_page,
        #             content_type=PageContentType.MARKDOWN,
        #         )
        #     )

        # return parsed_pages

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
