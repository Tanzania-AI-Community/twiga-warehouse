from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.application.factories.chunker_factory import ChunkerFactory
from src.config.settings import settings
from src.domain.entities.book import BookConfig, ClassConfig, ResourceConfig, SubjectConfig
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, ChunkerConfig, ChunkerType
from src.domain.entities.table_of_contents import (
    TableOfContents,
    TableOfContentsParserConfig,
    TableOfContentsParserType,
)
from src.infrastructure.table_of_contents.table_of_contents import get_table_of_contents


DEFAULT_LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"


def parse_comma_separated_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a comma-separated list of integers") from exc


def normalize_page_numbers(value: int | str | list[int]) -> int | list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return parse_comma_separated_ints(value)
    raise ValueError(f"Unsupported page number format: {value!r}")


def derive_input_file_name(subject_name: str, form: str) -> str:
    return f"{subject_name}_{form}.pdf"


def resolve_book_paths(
    input_dir: Path,
    input_file_name: str,
    output_file_name: str,
    input_root: Path | None = None,
    output_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    books_root = input_root or Path(settings.INPUT_BOOKS_PATH)
    output_root = output_root or Path(settings.OUTPUT_BOOKS_PATH)

    book_dir = books_root / input_dir
    info_path = book_dir / "info.yaml"
    input_path = book_dir / input_file_name
    output_path = output_root / output_file_name

    return info_path, input_path, output_path


def load_info_yaml(info_path: Path) -> dict[str, Any]:
    with info_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_resource_class_and_subject_config(
    yaml_data: dict[str, Any],
) -> tuple[ResourceConfig, ClassConfig, SubjectConfig]:
    resource_config = ResourceConfig(
        name=yaml_data["resource"].get("name", ""),
        type=yaml_data["resource"].get("type", "textbook"),
        authors=yaml_data["resource"].get("authors", list()),
    )

    class_config = ClassConfig(
        name=yaml_data["class"].get("name", ""),
        grade_level=yaml_data["class"].get("grade_level", ""),
        status=yaml_data["class"].get("status", ""),
    )

    subject_config = SubjectConfig(name=yaml_data["subject"].get("name", ""))

    return resource_config, class_config, subject_config


def build_book_config(
    info_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    chunker_type: ChunkerType,
    llm_model_name: str | None = None,
    embedding_model_name: str | None = None,
    page_batch_size: int | None = None,
) -> BookConfig:
    yaml_data = load_info_yaml(info_path)
    resolved_llm_model = llm_model_name or DEFAULT_LLM_MODEL
    resolved_embedding_model = embedding_model_name or DEFAULT_EMBEDDING_MODEL

    chunker_config = ChunkerConfig(
        chunker_type=chunker_type,
        last_page_number=yaml_data["book_config"].get("last_page_number"),
        page_batch_size=page_batch_size,
        llm_model_name=resolved_llm_model,
        embedding_model_name=resolved_embedding_model,
    )

    resource_config, class_config, subject_config = get_resource_class_and_subject_config(yaml_data)

    toc_parser_config = TableOfContentsParserConfig(
        parser_type=yaml_data["book_config"].get(
            "table_of_contents_parser", TableOfContentsParserType.OLLAMA
        )
    )

    return BookConfig(
        input_path=input_path,
        output_path=output_path,
        resource=resource_config,
        class_=class_config,
        subject=subject_config,
        chunker_config=chunker_config,
        table_of_contents_page_number=normalize_page_numbers(
            yaml_data["book_config"]["table_of_contents_page_number"]
        ),
        table_of_contents_parser=toc_parser_config,
        first_page_number=yaml_data["book_config"]["first_page_number"],
    )


def create_output_payload(
    config: BookConfig,
    chunks: list[Chunk],
    table_of_contents: TableOfContents,
) -> dict[str, Any]:
    return {
        "resource": config.resource.model_dump(),
        "class": config.class_.model_dump(),
        "subject": config.subject.model_dump(),
        "table_of_contents": table_of_contents.model_dump(),
        "chunker_config": config.chunker_config.model_dump(),
        "chunks": [chunk.model_dump() for chunk in chunks],
    }


def run_pipeline(config: BookConfig) -> dict[str, Any]:
    chunker_factory = ChunkerFactory(config.chunker_config)
    chunker: Chunker = chunker_factory.get_chunker()

    toc: TableOfContents = get_table_of_contents(
        pdf_path=config.input_path,
        toc_page_number=config.table_of_contents_page_number,
        parser_config=config.table_of_contents_parser,
    )

    logging.info("Table of contents:\n%s", toc)

    chunks: list[Chunk] = chunker.chunk(
        book_path=config.input_path,
        table_of_contents=toc,
        text_initial_page=config.first_page_number,
    )

    return create_output_payload(config=config, chunks=chunks, table_of_contents=toc)


def write_output(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)


def ensure_ocr_pdf(
    input_path: Path,
    output_path: Path | None = None,
    *,
    skip_text: bool = True,
) -> Path:
    resolved_output = output_path or input_path.with_name(
        f"{input_path.stem}_ocr{input_path.suffix}"
    )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    command = ["ocrmypdf"]
    if skip_text:
        command.append("--skip-text")

    command.extend([str(input_path), str(resolved_output)])

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ocrmypdf is not installed or not available in PATH. "
            "Install it from https://github.com/ocrmypdf/OCRmyPDF."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ocrmypdf failed with exit code {exc.returncode}") from exc

    return resolved_output
