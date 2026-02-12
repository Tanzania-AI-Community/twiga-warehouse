import argparse
from pathlib import Path

from src.application.pipeline_runner import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    build_book_config,
    resolve_book_paths,
    run_pipeline,
    write_output,
)
from src.domain.entities.chunker import ChunkerType


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunker_type",
        type=ChunkerType,
        required=True,
        choices=[
            ChunkerType.LANGCHAIN,
            ChunkerType.MATHEMATICAL,
        ],
        help="Specify which chunker to use (langchain or mathematical).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Relative path under INPUT_BOOKS_PATH containing the PDF and info.yaml.",
    )
    parser.add_argument(
        "--input_file_name",
        type=str,
        required=True,
        help="Filename of the PDF to be chunked.",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        required=True,
        help="Filename of the output JSON to write under OUTPUT_BOOKS_PATH.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        required=False,
        default=DEFAULT_LLM_MODEL,
        help="Model to use if using the LLM-based chunker.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=False,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model used for chunk embeddings.",
    )
    parser.add_argument(
        "--page_batch_size",
        type=int,
        default=None,
        required=False,
        help="Batch size of chunked pages when using the LLM chunker.",
    )

    args = parser.parse_args()

    info_path, input_path, output_path = resolve_book_paths(
        Path(args.input_dir),
        args.input_file_name,
        args.output_file_name,
    )

    config = build_book_config(
        info_path=info_path,
        input_path=input_path,
        output_path=output_path,
        chunker_type=args.chunker_type,
        llm_model_name=args.llm_model,
        embedding_model_name=args.embedding_model,
        page_batch_size=args.page_batch_size,
    )

    output_payload = run_pipeline(config)
    write_output(config.output_path, output_payload)


if __name__ == "__main__":
    main()
