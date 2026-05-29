import json
from pathlib import Path
from logging import getLogger

from src.models import (
    ChunkerType,
    EmbeddedChunk,
    EmbedderProvider,
    ParserType,
    PipelineRequest,
    TableOfContents,
    TableOfContentsParserConfig,
    TableOfContentsParserType,
    TextChunk,
)
from src.pipeline.ocr import ensure_ocr_pdf
from src.providers.chunker import LangchainChunker, MathematicalChunker
from src.providers.embedder import get_embedding_client
from src.providers.parser import HostedParser, MistralOcrParser, PdfTextParser
from src.providers.toc import extract_table_of_contents

logging = getLogger(__name__)

TABLE_OF_CONTENTS_SAVE_NAME = "table_of_contents.json"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_EMBEDDING_PROVIDER = EmbedderProvider.TOGETHER
DEFAULT_TOC_PARSER_TYPE = TableOfContentsParserType.TOGETHER


def _prepare_checkpoints_path(checkpoints_path: Path) -> Path:
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    return checkpoints_path


def _load_saved_toc(checkpoints_path: Path) -> TableOfContents | None:
    toc_file = checkpoints_path / TABLE_OF_CONTENTS_SAVE_NAME
    if not toc_file.is_file():
        logging.warning(f"No saved table of contents found at {toc_file}. Proceeding with extraction.")
        return None

    with toc_file.open(mode="r", encoding="utf-8") as handle:
        data = json.load(handle)
    return TableOfContents.model_validate(data)


def _save_toc(toc: TableOfContents, checkpoints_path: Path) -> None:
    toc_file = checkpoints_path / TABLE_OF_CONTENTS_SAVE_NAME
    with toc_file.open(mode="w", encoding="utf-8") as handle:
        json.dump(
            obj=toc.model_dump(),
            fp=handle,
            ensure_ascii=False,
            indent=4,
        )


def _load_table_of_contents(
    request: PipelineRequest,
    resolved_input_path: Path,
    resolved_toc_parser_type: TableOfContentsParserType,
    checkpoints_path: Path,
) -> TableOfContents:
    table_of_contents = _load_saved_toc(checkpoints_path=checkpoints_path)

    if not table_of_contents or not table_of_contents.chapters:
        table_of_contents = extract_table_of_contents(
            pdf_path=resolved_input_path,
            toc_page_numbers=request.book.pagination.table_of_contents_page_numbers,
            parser_config=TableOfContentsParserConfig(
                parser_type=resolved_toc_parser_type,
            ),
        )

    if table_of_contents.chapters and checkpoints_path:
        _save_toc(toc=table_of_contents, checkpoints_path=checkpoints_path)


def run_pipeline(
    request: PipelineRequest,
) -> dict[str, object]:
    checkpoints_path = _prepare_checkpoints_path(checkpoints_path=request.book.source_paths.checkpoints_path)
    resolved_parser_type = resolve_parser_type(
        chunker_type=request.processing.chunker_type,
        parser_type=request.processing.parser_type,
    )
    resolved_input_path = prepare_input_path(request=request)
    resolved_toc_parser_type = resolve_toc_parser_type(
        parser_type=resolved_parser_type,
        toc_parser_type=request.processing.toc_parser_type,
    )

    table_of_contents = _load_table_of_contents(
        request=request,
        resolved_input_path=resolved_input_path,
        resolved_toc_parser_type=resolved_toc_parser_type,
        checkpoints_path=checkpoints_path,
    )

    parser = get_parser(parser_type=resolved_parser_type)

    parsed_document = parser.parse(
        pdf_path=resolved_input_path,
        table_of_contents=table_of_contents,
        first_page_number=request.book.pagination.first_page_number,
    )

    chunker = get_chunker(chunker_type=request.processing.chunker_type)
    text_chunks = chunker.chunk(
        parsed_document=parsed_document,
        table_of_contents=table_of_contents,
        first_page_number=request.book.pagination.first_page_number,
        last_page_number=request.book.pagination.last_page_number,
    )

    embedded_chunks = create_embedded_chunks(
        embedding_provider=request.processing.embedding_provider,
        embedding_model_name=request.processing.embedding_model_name or DEFAULT_EMBEDDING_MODEL,
        text_chunks=text_chunks,
    )

    resolved_request = request.model_copy(
        update={
            "processing": request.processing.model_copy(
                update={
                    "parser_type": resolved_parser_type,
                    "toc_parser_type": resolved_toc_parser_type,
                    "embedding_model_name": request.processing.embedding_model_name
                    or DEFAULT_EMBEDDING_MODEL,
                }
            )
        }
    )
    return build_output_payload(
        request=resolved_request,
        table_of_contents=table_of_contents,
        embedded_chunks=embedded_chunks,
    )


def run_and_write_pipeline(
    request: PipelineRequest,
) -> Path:
    payload = run_pipeline(request=request)
    output_path = request.book.source_paths.output_path
    write_output(
        output_path=output_path,
        payload=payload,
    )
    return output_path


def prepare_input_path(
    request: PipelineRequest,
) -> Path:
    input_path = request.book.source_paths.input_path

    if not request.runtime.ocr_pdf:
        return input_path

    ocr_output_path = None
    if request.runtime.ocr_output_file_name:
        ocr_output_path = input_path.with_name(request.runtime.ocr_output_file_name)

    return ensure_ocr_pdf(
        input_path=input_path,
        output_path=ocr_output_path,
    )


def resolve_parser_type(
    chunker_type: ChunkerType,
    parser_type: ParserType | None,
) -> ParserType:
    if parser_type is not None:
        return parser_type

    if chunker_type == ChunkerType.MATHEMATICAL:
        return ParserType.MISTRAL

    return ParserType.PDF


def resolve_toc_parser_type(
    parser_type: ParserType,
    toc_parser_type: TableOfContentsParserType,
) -> TableOfContentsParserType:
    if toc_parser_type == TableOfContentsParserType.NONE:
        return toc_parser_type

    if parser_type == ParserType.HOSTED:
        return TableOfContentsParserType.HOSTED

    return toc_parser_type


def get_parser(
    parser_type: ParserType,
):
    if parser_type == ParserType.HOSTED:
        return HostedParser()

    if parser_type == ParserType.MISTRAL:
        return MistralOcrParser()

    return PdfTextParser()


def get_chunker(
    chunker_type: ChunkerType,
):
    if chunker_type == ChunkerType.MATHEMATICAL:
        return MathematicalChunker()

    if chunker_type == ChunkerType.LANGCHAIN:
        return LangchainChunker()

    raise ValueError(f"Unsupported chunker type: {chunker_type}")


def create_embedded_chunks(
    embedding_provider: EmbedderProvider,
    embedding_model_name: str,
    text_chunks: list[TextChunk],
) -> list[EmbeddedChunk]:
    if not text_chunks:
        raise ValueError("No chunks were produced for the selected book and chunker.")

    embedder = get_embedding_client(
        provider=embedding_provider,
        model_name=embedding_model_name,
    )
    embeddings = embedder.embed_documents(
        texts=[chunk.content for chunk in text_chunks],
    )

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk, embedding in zip(text_chunks, embeddings):
        if not embedding:
            continue

        embedded_chunks.append(
            EmbeddedChunk(
                content=chunk.content,
                page_number=chunk.page_number,
                chapter_number=chunk.chapter_number,
                embedding=embedding,
            )
        )

    if not embedded_chunks:
        raise ValueError("No embeddings were generated for the produced chunks.")

    return embedded_chunks


def build_output_payload(
    request: PipelineRequest,
    table_of_contents: TableOfContents,
    embedded_chunks: list[EmbeddedChunk],
) -> dict[str, object]:
    return {
        "resource": request.book.metadata.resource.model_dump(),
        "class": request.book.metadata.class_.model_dump(),
        "subject": request.book.metadata.subject.model_dump(),
        "table_of_contents": table_of_contents.model_dump(),
        "processing": request.processing.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "chunks": [chunk.model_dump() for chunk in embedded_chunks],
    }


def write_output(
    output_path: Path,
    payload: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8") as handle:
        json.dump(
            obj=payload,
            fp=handle,
            ensure_ascii=False,
            indent=4,
        )
