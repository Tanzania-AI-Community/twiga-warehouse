from pathlib import Path

from pypdf import PdfWriter

from src.models.document import ParsedDocument
from src.models.toc import Chapter, TableOfContents
from src.providers.parser.hosted_parser import HostedParser


def _write_pdf(path: Path, page_count: int) -> None:
    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)

    with path.open("wb") as handle:
        writer.write(handle)


def test_hosted_parser_parses_chapters_with_pdf_page_offsets(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    _write_pdf(path=pdf_path, page_count=8)

    parser = HostedParser(base_url="https://example.com")
    captured_ranges: list[tuple[int, int, int, int]] = []

    def _fake_request_chapter_parse(
        *,
        page_data_urls: list[str],
        pdf_start_page: int,
        pdf_end_page: int,
        chapter: Chapter | None,
    ):
        captured_ranges.append((pdf_start_page, pdf_end_page, len(page_data_urls), chapter.number if chapter else 0))

        return {
            "pages": [
                {"page_number": page_number, "markdown": f"chapter-{chapter.number}-page-{page_number}"}
                for page_number in range(1, len(page_data_urls) + 1)
            ]
        }

    monkeypatch.setattr(parser, "_request_chapter_parse", _fake_request_chapter_parse)

    parsed_document = parser.parse(
        pdf_path=pdf_path,
        table_of_contents=TableOfContents(
            chapters=[
                Chapter(name="Chapter 1", number=1, start_page=1),
                Chapter(name="Chapter 2", number=2, start_page=4),
            ]
        ),
        first_page_number=2,
    )

    assert captured_ranges == [(2, 4, 3, 1), (5, 8, 4, 2)]
    assert parsed_document == ParsedDocument(
        pages=[
            {"page_number": 2, "text": "chapter-1-page-1\n"},
            {"page_number": 3, "text": "chapter-1-page-2\n"},
            {"page_number": 4, "text": "chapter-1-page-3\n"},
            {"page_number": 5, "text": "chapter-2-page-1\n"},
            {"page_number": 6, "text": "chapter-2-page-2\n"},
            {"page_number": 7, "text": "chapter-2-page-3\n"},
            {"page_number": 8, "text": "chapter-2-page-4\n"},
        ]
    )


def test_hosted_parser_falls_back_to_entire_document_without_toc(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    _write_pdf(path=pdf_path, page_count=3)

    parser = HostedParser(base_url="https://example.com")

    def _fake_request_chapter_parse(
        *,
        page_data_urls: list[str],
        pdf_start_page: int,
        pdf_end_page: int,
        chapter: Chapter | None,
    ):
        del chapter
        assert (pdf_start_page, pdf_end_page) == (1, 3)
        assert len(page_data_urls) == 3
        return {"pages": ["page-1", "page-2", "page-3"]}

    monkeypatch.setattr(parser, "_request_chapter_parse", _fake_request_chapter_parse)

    parsed_document = parser.parse(pdf_path=pdf_path)

    assert [page.page_number for page in parsed_document.pages] == [1, 2, 3]
    assert [page.text for page in parsed_document.pages] == ["page-1\n", "page-2\n", "page-3\n"]


def test_hosted_parser_decodes_fenced_json_response() -> None:
    payload = HostedParser._parse_response_payload(
        response_body='''```json
{"pages":[{"page_number":1,"markdown":"hello"}]}
```'''
    )

    assert payload == {"pages": [{"page_number": 1, "markdown": "hello"}]}


def test_hosted_parser_accepts_string_page_numbers() -> None:
    parsed_pages = HostedParser._response_to_parsed_pages(
        response_payload={"pages": [{"page_number": "2", "markdown": "hello"}]},
        pdf_start_page=5,
        pdf_end_page=6,
    )

    assert [page.page_number for page in parsed_pages] == [6]
    assert [page.text for page in parsed_pages] == ["hello\n"]
