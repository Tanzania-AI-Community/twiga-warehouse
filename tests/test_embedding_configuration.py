from pathlib import Path

from src.application.pipeline_runner import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    build_book_config,
)
from src.domain.entities.chunker import ChunkerType, EmbedderProvider
from src.domain.entities.table_of_contents import TableOfContentsParserType


def _write_info_yaml(path: Path) -> None:
    path.write_text(
        """
resource:
  name: "Sample Book"
  type: "textbook"
  authors:
    - "Author One"

subject:
  name: "biology"

class:
  grade_level: "os4"
  status: "active"
  name: "Biology Form 4"

book_config:
  table_of_contents_page_number: "4,5"
  first_page_number: 10
  last_page_number: 200
""".strip(),
        encoding="utf-8",
    )


def test_build_book_config_uses_default_embedding_settings(tmp_path: Path) -> None:
    info_path = tmp_path / "info.yaml"
    _write_info_yaml(info_path)

    output = build_book_config(
        info_path=info_path,
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.json",
        chunker_type=ChunkerType.LANGCHAIN,
    )

    assert output.chunker_config.embedding_provider == DEFAULT_EMBEDDING_PROVIDER
    assert output.chunker_config.embedding_model_name == DEFAULT_EMBEDDING_MODEL
    if DEFAULT_EMBEDDING_PROVIDER == EmbedderProvider.TOGETHER:
        assert output.table_of_contents_parser.parser_type == TableOfContentsParserType.TOGETHER
    else:
        assert output.table_of_contents_parser.parser_type == TableOfContentsParserType.OLLAMA


def test_build_book_config_allows_together_embedding_provider(tmp_path: Path) -> None:
    info_path = tmp_path / "info.yaml"
    _write_info_yaml(info_path)

    output = build_book_config(
        info_path=info_path,
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.json",
        chunker_type=ChunkerType.LANGCHAIN,
        embedding_provider=EmbedderProvider.TOGETHER,
        embedding_model_name="intfloat/multilingual-e5-large-instruct",
    )

    assert output.chunker_config.embedding_provider == EmbedderProvider.TOGETHER
    assert output.chunker_config.embedding_model_name == "intfloat/multilingual-e5-large-instruct"
    assert output.table_of_contents_parser.parser_type == TableOfContentsParserType.TOGETHER


def test_build_book_config_defaults_toc_parser_to_ollama_for_ollama_provider(tmp_path: Path) -> None:
    info_path = tmp_path / "info.yaml"
    _write_info_yaml(info_path)

    output = build_book_config(
        info_path=info_path,
        input_path=tmp_path / "input.pdf",
        output_path=tmp_path / "output.json",
        chunker_type=ChunkerType.LANGCHAIN,
        embedding_provider=EmbedderProvider.OLLAMA,
    )

    assert output.table_of_contents_parser.parser_type == TableOfContentsParserType.OLLAMA
