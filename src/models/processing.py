from enum import Enum


class ParserType(str, Enum):
    PDF = "pdf"
    MISTRAL = "mistral"
    HOSTED = "hosted"


class ChunkerType(str, Enum):
    LANGCHAIN = "langchain"
    MATHEMATICAL = "mathematical"


class EmbedderProvider(str, Enum):
    OLLAMA = "ollama"
    TOGETHER = "together"
