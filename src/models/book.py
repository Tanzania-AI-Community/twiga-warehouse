from pathlib import Path

from pydantic import BaseModel, Field

from src.models.metadata import ClassMetadata, ResourceMetadata, SubjectMetadata
from src.models.processing import ChunkerType, EmbedderProvider, ParserType
from src.models.toc import TableOfContentsParserType


class BookMetadata(BaseModel):
    resource: ResourceMetadata
    class_: ClassMetadata
    subject: SubjectMetadata


class BookSourcePaths(BaseModel):
    input_dir: Path
    info_path: Path
    input_path: Path
    output_path: Path
    checkpoints_path: Path


class BookPagination(BaseModel):
    table_of_contents_page_numbers: int | list[int]
    first_page_number: int
    last_page_number: int | None = None


class BookDefinition(BaseModel):
    metadata: BookMetadata
    source_paths: BookSourcePaths
    pagination: BookPagination


class ProcessingOptions(BaseModel):
    chunker_type: ChunkerType
    parser_type: ParserType | None = None
    toc_parser_type: TableOfContentsParserType = TableOfContentsParserType.TOGETHER
    embedding_provider: EmbedderProvider = EmbedderProvider.TOGETHER
    embedding_model_name: str | None = None


class RuntimeOptions(BaseModel):
    ocr_pdf: bool = False
    ocr_output_file_name: str | None = None


class PipelineRequest(BaseModel):
    book: BookDefinition
    processing: ProcessingOptions
    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)
