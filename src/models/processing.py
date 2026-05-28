from enum import Enum


class ParserType(str, Enum):
    PDF = "pdf"
    MISTRAL = "mistral"


class ChunkerType(str, Enum):
    LANGCHAIN = "langchain"
    MATHEMATICAL = "mathematical"


class EmbedderProvider(str, Enum):
    OLLAMA = "ollama"
    TOGETHER = "together"
