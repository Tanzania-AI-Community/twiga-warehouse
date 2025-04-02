import argparse
import json
from typing import Any

from pydantic import BaseModel

from src.application.factories.chunker_factory import ChunkerFactory
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, ChunkerConfig, ChunkerType


class BookConfig(BaseModel):
    input_path: str
    output_path: str
    title: str
    author: str
    chunker_config: ChunkerConfig


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
    parser.add_argument("--author", type=str, required=True, help="Path to the input book to be chunked.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input book to be chunked.")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the chunked output.")

    args = parser.parse_args()
    
    config = create_config(args)
    chunker_factory = ChunkerFactory(config.chunker_config)

    chunker: Chunker = chunker_factory.get_chunker()

    chunks: list[Chunk] = chunker.chunk(book_path=config.input_path)

    output_file = create_output_file(config=config, chunks=chunks)

    output_path = f"./{config.output_path}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_file, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
