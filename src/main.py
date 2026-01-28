import argparse
import json
from typing import Any
import yaml

import logging

from src.application.factories.chunker_factory import ChunkerFactory
from src.config.settings import settings
from src.domain.entities.book import BookConfig, ClassConfig, ChunkerConfig, SubjectConfig, ResourceConfig
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, ChunkerConfig, ChunkerType
from src.domain.entities.table_of_contents import (
    TableOfContents,
    TableOfContentsParserConfig,
    TableOfContentsParserType,
)
from src.infrastructure.table_of_contents.table_of_contents import get_table_of_contents


def comma_separated_ints(s: str) -> list[int]:
    try:
        return [int(item) for item in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{s}' is not a comma-separated list of integers")


def get_resource_class_and_subject_config(yaml_data: dict[str, Any]) -> tuple[ResourceConfig, ClassConfig, SubjectConfig]:
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


def create_config(args) -> BookConfig:
    yaml_file_path = settings.INPUT_BOOKS_PATH + args.input_dir + "info.yaml"
    with open(yaml_file_path) as f:
        yaml_data = yaml.safe_load(f)

    chunker_config = ChunkerConfig(
        chunker_type=args.chunker_type,
        last_page_number=yaml_data["book_config"].get("last_page_number"),
        page_batch_size=args.page_batch_size,
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model,
    )

    resource_config, class_config, subject_config = get_resource_class_and_subject_config(yaml_data)

    toc_parser_config = TableOfContentsParserConfig(
        parser_type=yaml_data["book_config"].get(
            "table_of_contents_parser", TableOfContentsParserType.OLLAMA
        )
    )

    return BookConfig(
        input_path=settings.INPUT_BOOKS_PATH + args.input_dir + args.input_file_name,
        output_path=settings.OUTPUT_BOOKS_PATH + args.output_file_name,
        resource=resource_config,
        class_=class_config,
        subject=subject_config,
        chunker_config=chunker_config,
        table_of_contents_page_number=comma_separated_ints(yaml_data["book_config"]["table_of_contents_page_number"]),
        table_of_contents_parser=toc_parser_config,
        first_page_number=yaml_data["book_config"]["first_page_number"],
    )


def create_output_file(
    config: BookConfig, chunks: list[Chunk], table_of_contents: TableOfContents
) -> dict[str, Any]:
    return {
        "resource": config.resource.model_dump(),
        "class": config.class_.model_dump(),
        "subject": config.subject.model_dump(),
        "table_of_contents": table_of_contents.model_dump(),
        "chunker_config": config.chunker_config.model_dump(),
        "chunks": [chunk.model_dump() for chunk in chunks],
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunker_type",
        type=ChunkerType,
        required=True,
        choices=[
            ChunkerType.LANGCHAIN, ChunkerType.LLM, ChunkerType.MATHEMATICAL
        ],
        help="Specify which chunker to use (langchain or unstructured).",
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input book and general info to be chunked.")
    parser.add_argument("--input_file_name", type=str, required=True, help="Filename of the PDF to be chunked.")
    parser.add_argument("--output_file_name", type=str, required=True, help="Path to save the chunked output.")
    parser.add_argument(
        "--llm_model",
        type=str,
        required=False,
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        help="Model to use if using LLMChunker.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=False,
        default="BAAI/bge-large-en-v1.5",
        help="Model to calculate the embedding of the text",
    )
    parser.add_argument(
        "--page_batch_size",
        type=int,
        default=None,
        required=False,
        help="Batch size of chunked pages when using LLMChunker.",
    )

    args = parser.parse_args()
    config = create_config(args)

    chunker_factory = ChunkerFactory(config.chunker_config)
    chunker: Chunker = chunker_factory.get_chunker()

    toc: TableOfContents = get_table_of_contents(
        pdf_path=config.input_path,
        toc_page_number=config.table_of_contents_page_number,
        parser_config=config.table_of_contents_parser,
    )

    logging.info(f"Table of contents:\n{toc}")

    chunks: list[Chunk] = chunker.chunk(book_path=config.input_path, table_of_contents=toc, text_initial_page=config.first_page_number)

    output_file = create_output_file(config=config, chunks=chunks, table_of_contents=toc)

    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(output_file, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
