from pydantic import BaseModel


class ResourceMetadata(BaseModel):
    name: str
    type: str
    authors: list[str]


class ClassMetadata(BaseModel):
    grade_level: str
    name: str
    status: str


class SubjectMetadata(BaseModel):
    name: str
