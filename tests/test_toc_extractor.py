from pathlib import Path

from pypdf import PdfWriter

from src.models.toc import Chapter, TableOfContents, TableOfContentsParserConfig, TableOfContentsParserType
from src.providers.toc.llm_toc_extractor import (
    TOC_EXTRACTION_TEMPLATE,
    extract_table_of_contents,
)


def _write_pdf(path: Path, page_count: int) -> None:
    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)

    with path.open("wb") as handle:
        writer.write(handle)


def test_extract_table_of_contents_uses_hosted_model_with_selected_pages(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    _write_pdf(path=pdf_path, page_count=5)
    captured: dict[str, object] = {}

    class _FakeNuExtractClient:
        def render_pdf_pages_to_data_urls(self, *, pdf_path: Path, page_numbers: list[int]) -> list[str]:
            captured["pdf_path"] = pdf_path
            captured["page_numbers"] = page_numbers
            return [f"page-{page_number}" for page_number in page_numbers]

        def extract_structured(
            self,
            *,
            page_data_urls: list[str],
            template: dict[str, object],
            instructions: str | None = None,
            temperature: float | None = None,
        ) -> object:
            captured["page_data_urls"] = page_data_urls
            captured["template"] = template
            captured["instructions"] = instructions
            captured["temperature"] = temperature
            return {
                "chapters": [
                    {"name": "Waves", "number": 1, "start_page": 1},
                ]
            }

    monkeypatch.setattr("src.providers.toc.llm_toc_extractor.NuExtractClient", _FakeNuExtractClient)

    table_of_contents = extract_table_of_contents(
        pdf_path=pdf_path,
        toc_page_numbers=[2, 4],
        parser_config=TableOfContentsParserConfig(parser_type=TableOfContentsParserType.HOSTED),
    )

    assert captured["pdf_path"] == pdf_path
    assert captured["page_numbers"] == [2, 4]
    assert captured["page_data_urls"] == ["page-2", "page-4"]
    assert captured["template"] == TOC_EXTRACTION_TEMPLATE
    assert "glossary" in str(captured["instructions"]).lower()
    assert captured["temperature"] == 0
    assert table_of_contents == TableOfContents(
        chapters=[Chapter(name="Waves", number=1, start_page=1)]
    )


def test_extract_table_of_contents_normalizes_hosted_model_output(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    _write_pdf(path=pdf_path, page_count=2)

    class _FakeNuExtractClient:
        def render_pdf_pages_to_data_urls(self, *, pdf_path: Path, page_numbers: list[int]) -> list[str]:
            del pdf_path, page_numbers
            return ["page-1"]

        def extract_structured(
            self,
            *,
            page_data_urls: list[str],
            template: dict[str, object],
            instructions: str | None = None,
            temperature: float | None = None,
        ) -> object:
            del page_data_urls, template, instructions, temperature
            return {
                "chapters": [
                    {"name": "Waves", "number": "1", "start_page": "7"},
                    {"name": None, "number": 2, "start_page": 15},
                ]
            }

    monkeypatch.setattr("src.providers.toc.llm_toc_extractor.NuExtractClient", _FakeNuExtractClient)

    table_of_contents = extract_table_of_contents(
        pdf_path=pdf_path,
        toc_page_numbers=1,
        parser_config=TableOfContentsParserConfig(parser_type=TableOfContentsParserType.HOSTED),
    )

    assert table_of_contents == TableOfContents(
        chapters=[Chapter(name="Waves", number=1, start_page=7)]
    )
