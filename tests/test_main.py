from pathlib import Path

from src.main import main
from src.models.book import (
    BookDefinition,
    BookMetadata,
    BookPagination,
    BookSourcePaths,
)
from src.models.metadata import ClassMetadata, ResourceMetadata, SubjectMetadata
from src.models.processing import ParserType


def _build_book_definition(tmp_path: Path) -> BookDefinition:
    pdf_path = tmp_path / "book.pdf"
    return BookDefinition(
        metadata=BookMetadata(
            resource=ResourceMetadata(name="Book", type="textbook", authors=["Author"]),
            class_=ClassMetadata(grade_level="os1", status="active", name="Class"),
            subject=SubjectMetadata(name="history"),
        ),
        source_paths=BookSourcePaths(
            input_dir=tmp_path,
            info_path=tmp_path / "info.yaml",
            input_path=pdf_path,
            output_path=tmp_path / "output.json",
            checkpoints_path=tmp_path / "checkpoints",
        ),
        pagination=BookPagination(
            table_of_contents_page_numbers=1,
            first_page_number=1,
        ),
    )


def test_main_defaults_parser_type_to_pdf(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "src.main",
            "--chunker_type",
            "langchain",
            "--input_dir",
            "form_1/history",
            "--input_file_name",
            "book.pdf",
            "--output_file_name",
            "book.json",
        ],
    )
    monkeypatch.setattr("src.main.resolve_book_paths", lambda **kwargs: object())
    monkeypatch.setattr("src.main.build_book_definition", lambda paths: _build_book_definition(tmp_path))
    monkeypatch.setattr(
        "src.main.run_and_write_pipeline",
        lambda request: captured.setdefault("request", request),
    )

    main()

    assert captured["request"].processing.parser_type == ParserType.PDF
