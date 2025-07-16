import argparse
import json
from typing import Any, Union

import logging
from pydantic import BaseModel

from src.application.factories.chunker_factory import ChunkerFactory
from src.config.settings import settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, ChunkerConfig, ChunkerType
from src.domain.entities.table_of_contents import get_table_of_contents, TableOfContents


def comma_separated_ints(s: str) -> list[int]:
    try:
        return [int(item) for item in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{s}' is not a comma-separated list of integers")


def parse_toc_pages(value: str) -> Union[int, list[int]]:
    try:
        # Try parsing as a single integer first
        return int(value)
    except ValueError:
        # If not a single integer, try parsing as a list of integers
        try:
            # Remove brackets and split by commas
            values = value.strip('[]').split(',')
            return [int(v.strip()) for v in values]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"'{value}' must be either an integer or a comma-separated list of integers"
            )


class BookConfig(BaseModel):
    input_path: str
    output_path: str
    title: str
    author: str
    chunker_config: ChunkerConfig
    table_of_contents_page_number: Union[int|list[int]]
    first_page_number: int


def create_config(args) -> BookConfig:
    chunker_config = ChunkerConfig(
        chunker_type=args.chunker_type,
        last_page_number=args.last_page_number,
        page_batch_size=args.page_batch_size,
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model,
    )

    return BookConfig(
        input_path=settings.INPUT_BOOKS_PATH + args.input_file_name,
        output_path=settings.OUTPUT_BOOKS_PATH + args.output_file_name,
        title=args.title,
        author=args.author,
        chunker_config=chunker_config,
        table_of_contents_page_number=args.table_of_contents_page_number,
        first_page_number=args.first_page_number,
    )


def create_output_file(config: BookConfig, chunks: list[Chunk]) -> dict[str, Any]:
    return {
        "title": config.title,
        "author": config.author,
        "chunks": [chunk.model_dump() for chunk in chunks],
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunker_type",
        type=ChunkerType,
        required=True,
        choices=[
            ChunkerType.LANGCHAIN, ChunkerType.UNSTRUCTURED, ChunkerType.LLM
        ],
        help="Specify which chunker to use (langchain or unstructured).",
    )
    parser.add_argument("--title", type=str, required=True, help="Path to the input book to be chunked.")
    parser.add_argument(
        "--table_of_contents_page_number",
        type=comma_separated_ints,
        default=[],
        required=True, 
        help="Page number of the table of contents. Can be a single number or a comma-separated list (e.g., 5 or [4,5,6])",
    )
    parser.add_argument("--first_page_number", type=int, required=True, help="Page number of the first page.")
    parser.add_argument("--last_page_number", type=int, required=False, help="Page number of the last page.")
    parser.add_argument("--author", type=str, required=True, help="Path to the input book to be chunked.")
    parser.add_argument("--input_file_name", type=str, required=True, help="Path to the input book to be chunked.")
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
        required=False,
        help="Batch size of chunked pages when using LLMChunker.",
    )

    args = parser.parse_args()
    config = create_config(args)

    chunker_factory = ChunkerFactory(config.chunker_config)

    chunker: Chunker = chunker_factory.get_chunker()

    toc: TableOfContents = get_table_of_contents(
        pdf_path=config.input_path, toc_page_number=config.table_of_contents_page_number
    )

    logging.info(f"Table of contents:\n{toc}")

    chunks: list[Chunk] = chunker.chunk(book_path=config.input_path, table_of_contents=toc, text_initial_page=config.first_page_number)

    output_file = create_output_file(config=config, chunks=chunks)

    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(output_file, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
