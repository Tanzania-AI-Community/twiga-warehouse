from dataclasses import dataclass
from pathlib import Path

from dagster import Config, Definitions, job, op

from src.models import (
    BookSourcePaths,
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
    derive_input_file_name,
    resolve_book_paths,
    run_and_write_pipeline,
)


@dataclass
class PipelineParams:
    ocr_pdf: bool
    chunker_type: ChunkerType
    subject_name: str
    form: str
    output_file_name: str
    embedding_model: str
    embedding_provider: EmbedderProvider
    parser_type: ParserType | None
    toc_parser_type: TableOfContentsParserType
    input_file_name: str | None
    ocr_output_file_name: str | None


class PipelineRunConfig(Config):
    ocr_pdf: bool = False
    chunker_type: ChunkerType = ChunkerType.LANGCHAIN
    subject_name: str
    form: str
    output_file_name: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_provider: EmbedderProvider = DEFAULT_EMBEDDING_PROVIDER
    parser_type: ParserType | None = None
    toc_parser_type: TableOfContentsParserType = DEFAULT_TOC_PARSER_TYPE
    input_file_name: str | None = None
    ocr_output_file_name: str | None = None


@op
def collect_params(config: PipelineRunConfig) -> PipelineParams:
    return PipelineParams(
        ocr_pdf=config.ocr_pdf,
        chunker_type=config.chunker_type,
        subject_name=config.subject_name,
        form=config.form,
        output_file_name=config.output_file_name,
        embedding_model=config.embedding_model,
        embedding_provider=config.embedding_provider,
        parser_type=config.parser_type,
        toc_parser_type=config.toc_parser_type,
        input_file_name=config.input_file_name,
        ocr_output_file_name=config.ocr_output_file_name,
    )


@op
def resolve_paths(params: PipelineParams) -> BookSourcePaths:
    input_dir = Path(params.form) / params.subject_name
    input_file_name = params.input_file_name or derive_input_file_name(
        subject_name=params.subject_name,
        form=params.form,
    )

    paths = resolve_book_paths(
        input_dir=input_dir,
        input_file_name=input_file_name,
        output_file_name=params.output_file_name,
    )

    return paths


@op
def build_request_op(params: PipelineParams, paths: BookSourcePaths) -> PipelineRequest:
    book_definition = build_book_definition(paths=paths)
    processing_options = ProcessingOptions(
        chunker_type=params.chunker_type,
        parser_type=params.parser_type,
        toc_parser_type=params.toc_parser_type,
        embedding_provider=params.embedding_provider,
        embedding_model_name=params.embedding_model,
    )
    runtime_options = RuntimeOptions(
        ocr_pdf=params.ocr_pdf,
        ocr_output_file_name=params.ocr_output_file_name,
    )
    return PipelineRequest(
        book=book_definition,
        processing=processing_options,
        runtime=runtime_options,
    )


@op
def run_pipeline_op(request: PipelineRequest) -> str:
    output_path = run_and_write_pipeline(request=request)
    return str(output_path)


@job
def book_pipeline_job():
    params = collect_params()
    paths = resolve_paths(params)
    request = build_request_op(params, paths)
    run_pipeline_op(request)


defs = Definitions(jobs=[book_pipeline_job])
