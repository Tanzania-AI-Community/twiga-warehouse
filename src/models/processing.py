from enum import Enum


class ParserType(str, Enum):
    PDF = "pdf"
    MISTRAL = "mistral"


class ChunkerType(str, Enum):
    LANGCHAIN = "langchain"
    MATHEMATICAL = "mathematical"


class EmbedderProvider(str, Enum):
    GOOGLE = "google"
    OLLAMA = "ollama"
    TOGETHER = "together"
