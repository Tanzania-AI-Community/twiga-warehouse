from typing import Union

from pydantic import BaseModel

from src.domain.entities.chunker import ChunkerConfig


class ResourceConfig(BaseModel):
    name: str
    type: str
    authors: list[str]


class ClassConfig(BaseModel):
    grade_level: str
    name: str
    status: str


class SubjectConfig(BaseModel):
    name: str


class BookConfig(BaseModel):
    input_path: str
    output_path: str
    resource: ResourceConfig
    class_: ClassConfig
    subject: SubjectConfig
    chunker_config: ChunkerConfig
    table_of_contents_page_number: Union[int|list[int]]
    first_page_number: int
