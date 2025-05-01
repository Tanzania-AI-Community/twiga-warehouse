import argparse
import json
from typing import Any, Union

from pydantic import BaseModel

from src.application.factories.chunker_factory import ChunkerFactory
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, ChunkerConfig, ChunkerType
from src.domain.entities.toc import get_toc

import logging


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
    table_of_contents_page_number: int
    first_page_number: int


def create_config(args) -> BookConfig:
    chunker_config = ChunkerConfig(
        chunker_type=args.chunker_type,
    )
    return BookConfig(
        input_path=args.input_path,
        output_path=args.output_path,
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
        choices=[ChunkerType.LANGCHAIN, ChunkerType.UNSTRUCTURED],
        help="Specify which chunker to use (langchain or unstructured).",
    )
    parser.add_argument("--title", type=str, required=True, help="Path to the input book to be chunked.")
    parser.add_argument("--table_of_contents_page_number", type=int, required=True, 
                        help="Page number of the table of contents. Can be a single number or a comma-separated list (e.g., 5 or [4,5,6])")
    parser.add_argument("--first_page_number", type=int, required=True, help="Page number of the first page.")
    parser.add_argument("--author", type=str, required=True, help="Path to the input book to be chunked.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input book to be chunked.")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the chunked output.")

    args = parser.parse_args()
    
    config = create_config(args)
    chunker_factory = ChunkerFactory(config.chunker_config)

    chunker: Chunker = chunker_factory.get_chunker()

    toc: TableOfContents = get_toc(config.input_path, config.table_of_contents_page_number)

    logging.info(f"Table of contents:\n{toc}")

    chunks: list[Chunk] = chunker.chunk(book_path=config.input_path, toc=toc)

    output_file = create_output_file(config=config, chunks=chunks)

    output_path = f"./{config.output_path}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_file, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
