import argparse
from pathlib import Path

from src.models import (
    ChunkerType,
    EmbedderProvider,
    ParserType,
    PipelineRequest,
    ProcessingOptions,
    RuntimeOptions,
    TableOfContentsParserType,
)
from src.pipeline import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_TOC_PARSER_TYPE,
    build_book_definition,
    resolve_book_paths,
    run_and_write_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunker_type",
        type=str,
        required=True,
        choices=[
            ChunkerType.LANGCHAIN.value,
            ChunkerType.MATHEMATICAL.value,
        ],
        help="Specify which chunker to use (langchain or mathematical).",
    )
    parser.add_argument(
        "--parser_type",
        type=str,
        required=False,
        default=None,
        choices=[
            ParserType.PDF.value,
            ParserType.MISTRAL.value,
            ParserType.HOSTED.value,
        ],
        help="Optional parser override.",
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
        "--embedding_model",
        type=str,
        required=False,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used for chunk embeddings.",
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        required=False,
        default=DEFAULT_EMBEDDING_PROVIDER.value,
        choices=[
            EmbedderProvider.OLLAMA.value,
            EmbedderProvider.TOGETHER.value,
        ],
        help="Embedding provider to use (ollama or together).",
    )
    parser.add_argument(
        "--toc_parser_type",
        type=str,
        required=False,
        default=DEFAULT_TOC_PARSER_TYPE.value,
        choices=[
            TableOfContentsParserType.GEMINI.value,
            TableOfContentsParserType.TOGETHER.value,
            TableOfContentsParserType.OLLAMA.value,
            TableOfContentsParserType.HOSTED.value,
            TableOfContentsParserType.NONE.value,
        ],
        help="Parser to use for table of contents extraction.",
    )
    parser.add_argument(
        "--ocr_pdf",
        action="store_true",
        help="Run OCRmyPDF before parsing.",
    )
    parser.add_argument(
        "--ocr_output_file_name",
        type=str,
        required=False,
        default=None,
        help="Optional output filename for the OCR PDF.",
    )

    args = parser.parse_args()

    paths = resolve_book_paths(
        input_dir=Path(args.input_dir),
        input_file_name=args.input_file_name,
        output_file_name=args.output_file_name,
    )
    book_definition = build_book_definition(paths=paths)

    processing_options = ProcessingOptions(
        chunker_type=ChunkerType(args.chunker_type),
        parser_type=ParserType(args.parser_type) if args.parser_type else None,
        toc_parser_type=TableOfContentsParserType(args.toc_parser_type),
        embedding_provider=EmbedderProvider(args.embedding_provider),
        embedding_model_name=args.embedding_model,
    )
    runtime_options = RuntimeOptions(
        ocr_pdf=args.ocr_pdf,
        ocr_output_file_name=args.ocr_output_file_name,
    )
    request = PipelineRequest(
        book=book_definition,
        processing=processing_options,
        runtime=runtime_options,
    )
    run_and_write_pipeline(request=request)


if __name__ == "__main__":
    main()
