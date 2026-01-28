from dataclasses import dataclass
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


@dataclass
class PipelineParams:
    ocr_pdf: bool
    chunker_type: ChunkerType
    subject_name: str
    form: str
    output_file_name: str
    embedding_parser: str
    llm_model: str
    page_batch_size: int | None
    input_file_name: str | None
    ocr_output_file_name: str | None


@dataclass
class PathBundle:
    info_path: Path
    input_path: Path
    output_path: Path


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
def collect_params(config: PipelineRunConfig) -> PipelineParams:
    if config.chunker_type not in SUPPORTED_CHUNKERS:
        raise ValueError(
            f"Unsupported chunker_type '{config.chunker_type}'. "
            f"Use one of: {', '.join(SUPPORTED_CHUNKERS)}."
        )
    return PipelineParams(
        ocr_pdf=config.ocr_pdf,
        chunker_type=config.chunker_type,
        subject_name=config.subject_name,
        form=config.form,
        output_file_name=config.output_file_name,
        embedding_parser=config.embedding_parser,
        llm_model=config.llm_model,
        page_batch_size=config.page_batch_size,
        input_file_name=config.input_file_name,
        ocr_output_file_name=config.ocr_output_file_name,
    )


@op
def resolve_paths(params: PipelineParams) -> PathBundle:
    input_dir = Path(params.form) / params.subject_name
    input_file_name = params.input_file_name or derive_input_file_name(
        params.subject_name,
        params.form,
    )

    info_path, input_path, output_path = resolve_book_paths(
        input_dir=input_dir,
        input_file_name=input_file_name,
        output_file_name=params.output_file_name,
    )

    return PathBundle(
        info_path=info_path,
        input_path=input_path,
        output_path=output_path,
    )


@op
def maybe_run_ocr(params: PipelineParams, paths: PathBundle) -> Path:
    if not params.ocr_pdf:
        return paths.input_path

    ocr_output = None
    if params.ocr_output_file_name:
        ocr_output = paths.input_path.with_name(params.ocr_output_file_name)

    return ensure_ocr_pdf(paths.input_path, output_path=ocr_output)


@op
def build_config_op(params: PipelineParams, paths: PathBundle, input_path: Path):
    return build_book_config(
        info_path=paths.info_path,
        input_path=input_path,
        output_path=paths.output_path,
        chunker_type=params.chunker_type,
        llm_model_name=params.llm_model,
        embedding_model_name=params.embedding_parser,
        page_batch_size=params.page_batch_size,
    )


@op
def run_pipeline_op(book_config):
    return run_pipeline(book_config)


@op
def write_output_op(book_config, payload: dict) -> str:
    write_output(book_config.output_path, payload)
    return str(book_config.output_path)


@job
def book_pipeline_job():
    params = collect_params()
    paths = resolve_paths(params)
    input_path = maybe_run_ocr(params, paths)
    book_config = build_config_op(params, paths, input_path)
    payload = run_pipeline_op(book_config)
    write_output_op(book_config, payload)


defs = Definitions(jobs=[book_pipeline_job])
