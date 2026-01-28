from pathlib import Path

from dagster import Config, Definitions, job, op

from src.application.pipeline_runner import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    build_book_config,
    derive_input_file_name,
    ensure_ocr_pdf,
    resolve_book_paths,
    run_pipeline,
    write_output,
)
from src.domain.entities.chunker import ChunkerType


SUPPORTED_CHUNKERS = (ChunkerType.LANGCHAIN, ChunkerType.MATHEMATICAL)


class PipelineRunConfig(Config):
    ocr_pdf: bool = False
    chunker_type: ChunkerType = ChunkerType.LANGCHAIN
    subject_name: str
    form: str
    output_file_name: str
    embedding_parser: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    page_batch_size: int | None = None
    input_file_name: str | None = None
    ocr_output_file_name: str | None = None


@op
def run_book_pipeline(config: PipelineRunConfig) -> str:
    if config.chunker_type not in SUPPORTED_CHUNKERS:
        raise ValueError(
            f"Unsupported chunker_type '{config.chunker_type}'. "
            f"Use one of: {', '.join(SUPPORTED_CHUNKERS)}."
        )

    input_dir = Path(config.form) / config.subject_name
    input_file_name = config.input_file_name or derive_input_file_name(
        config.subject_name,
        config.form,
    )

    info_path, input_path, output_path = resolve_book_paths(
        input_dir=input_dir,
        input_file_name=input_file_name,
        output_file_name=config.output_file_name,
    )

    if config.ocr_pdf:
        ocr_output = None
        if config.ocr_output_file_name:
            ocr_output = input_path.with_name(config.ocr_output_file_name)
        input_path = ensure_ocr_pdf(input_path, output_path=ocr_output)

    book_config = build_book_config(
        info_path=info_path,
        input_path=input_path,
        output_path=output_path,
        chunker_type=config.chunker_type,
        llm_model_name=config.llm_model,
        embedding_model_name=config.embedding_parser,
        page_batch_size=config.page_batch_size,
    )

    output_payload = run_pipeline(book_config)
    write_output(book_config.output_path, output_payload)

    return str(book_config.output_path)


@job
def book_pipeline_job():
    run_book_pipeline()


defs = Definitions(jobs=[book_pipeline_job])
