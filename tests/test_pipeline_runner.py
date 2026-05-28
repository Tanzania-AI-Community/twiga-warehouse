from pathlib import Path

from src.models.book import (
    BookDefinition,
    BookMetadata,
    BookPagination,
    BookSourcePaths,
    PipelineRequest,
    ProcessingOptions,
)
from src.models.chunk import EmbeddedChunk, TextChunk
from src.models.document import ParsedDocument, ParsedPage
from src.models.metadata import ClassMetadata, ResourceMetadata, SubjectMetadata
from src.models.processing import ChunkerType, EmbedderProvider, ParserType
from src.models.toc import Chapter, TableOfContents, TableOfContentsParserType
from src.pipeline.runner import run_pipeline


def _build_request(pdf_path: Path) -> PipelineRequest:
    return PipelineRequest(
        book=BookDefinition(
            metadata=BookMetadata(
                resource=ResourceMetadata(name="Book", type="textbook", authors=["Author"]),
                class_=ClassMetadata(grade_level="os1", status="active", name="Class"),
                subject=SubjectMetadata(name="history"),
            ),
            source_paths=BookSourcePaths(
                input_dir=pdf_path.parent,
                info_path=pdf_path.parent / "info.yaml",
                input_path=pdf_path,
                output_path=pdf_path.parent / "output.json",
            ),
            pagination=BookPagination(
                table_of_contents_page_numbers=1,
                first_page_number=5,
            ),
        ),
        processing=ProcessingOptions(
            chunker_type=ChunkerType.LANGCHAIN,
            parser_type=ParserType.HOSTED,
            toc_parser_type=TableOfContentsParserType.NONE,
            embedding_provider=EmbedderProvider.OLLAMA,
            embedding_model_name="test-model",
        ),
    )


def test_run_pipeline_passes_toc_to_hosted_parser(monkeypatch, tmp_path: Path) -> None:
    request = _build_request(pdf_path=tmp_path / "book.pdf")
    expected_toc = TableOfContents(chapters=[Chapter(name="Chapter 1", number=1, start_page=1)])
    parser_calls: dict[str, object] = {}

    class _FakeParser:
        def parse(self, *, pdf_path: Path, table_of_contents: TableOfContents | None = None, first_page_number: int = 1):
            parser_calls["pdf_path"] = pdf_path
            parser_calls["table_of_contents"] = table_of_contents
            parser_calls["first_page_number"] = first_page_number
            return ParsedDocument(pages=[ParsedPage(page_number=5, text="chapter text")])

    class _FakeChunker:
        def chunk(self, *, parsed_document: ParsedDocument, table_of_contents: TableOfContents, first_page_number: int, last_page_number: int | None = None):
            assert parsed_document.pages[0].page_number == 5
            assert table_of_contents == expected_toc
            assert first_page_number == 5
            assert last_page_number is None
            return [TextChunk(content="chapter text", page_number=1, chapter_number=1)]

    monkeypatch.setattr("src.pipeline.runner.prepare_input_path", lambda request: request.book.source_paths.input_path)
    monkeypatch.setattr("src.pipeline.runner.extract_table_of_contents", lambda **kwargs: expected_toc)
    monkeypatch.setattr("src.pipeline.runner.get_parser", lambda parser_type: _FakeParser())
    monkeypatch.setattr("src.pipeline.runner.get_chunker", lambda chunker_type: _FakeChunker())
    monkeypatch.setattr(
        "src.pipeline.runner.create_embedded_chunks",
        lambda **kwargs: [
            EmbeddedChunk(content="chapter text", page_number=1, chapter_number=1, embedding=[0.1, 0.2])
        ],
    )

    payload = run_pipeline(request=request)

    assert parser_calls == {
        "pdf_path": request.book.source_paths.input_path,
        "table_of_contents": expected_toc,
        "first_page_number": 5,
    }
    assert payload["table_of_contents"] == expected_toc.model_dump()


def test_run_pipeline_uses_hosted_toc_parser_for_hosted_document_parser(monkeypatch, tmp_path: Path) -> None:
    request = _build_request(pdf_path=tmp_path / "book.pdf").model_copy(
        update={
            "processing": _build_request(pdf_path=tmp_path / "book.pdf").processing.model_copy(
                update={"toc_parser_type": TableOfContentsParserType.TOGETHER}
            )
        }
    )
    expected_toc = TableOfContents(chapters=[Chapter(name="Chapter 1", number=1, start_page=1)])
    captured: dict[str, object] = {}

    class _FakeParser:
        def parse(self, *, pdf_path: Path, table_of_contents: TableOfContents | None = None, first_page_number: int = 1):
            del pdf_path, table_of_contents, first_page_number
            return ParsedDocument(pages=[ParsedPage(page_number=5, text="chapter text")])

    class _FakeChunker:
        def chunk(self, *, parsed_document: ParsedDocument, table_of_contents: TableOfContents, first_page_number: int, last_page_number: int | None = None):
            del parsed_document, table_of_contents, first_page_number, last_page_number
            return [TextChunk(content="chapter text", page_number=1, chapter_number=1)]

    def _fake_extract_table_of_contents(**kwargs):
        captured["toc_parser_type"] = kwargs["parser_config"].parser_type
        return expected_toc

    monkeypatch.setattr("src.pipeline.runner.prepare_input_path", lambda request: request.book.source_paths.input_path)
    monkeypatch.setattr("src.pipeline.runner.extract_table_of_contents", _fake_extract_table_of_contents)
    monkeypatch.setattr("src.pipeline.runner.get_parser", lambda parser_type: _FakeParser())
    monkeypatch.setattr("src.pipeline.runner.get_chunker", lambda chunker_type: _FakeChunker())
    monkeypatch.setattr(
        "src.pipeline.runner.create_embedded_chunks",
        lambda **kwargs: [
            EmbeddedChunk(content="chapter text", page_number=1, chapter_number=1, embedding=[0.1, 0.2])
        ],
    )

    payload = run_pipeline(request=request)

    assert captured["toc_parser_type"] == TableOfContentsParserType.HOSTED
    assert payload["processing"]["toc_parser_type"] == TableOfContentsParserType.HOSTED.value
