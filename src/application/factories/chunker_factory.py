from src.domain.entities.chunker import Chunker, ChunkerConfig, ChunkerType
from src.infrastructure.chunker.langchain_chunker import LangchainChunker
from src.infrastructure.chunker.mathematical_chunker import MathematicalChunker


class ChunkerFactory:
    def __init__(self, config: ChunkerConfig):
        self.config = config

    def get_chunker(self) -> Chunker:
        if self.config.chunker_type == ChunkerType.UNSTRUCTURED:
            raise ValueError("Unstructured chunker is deprecated. Choose other.")
        
        if self.config.chunker_type == ChunkerType.LANGCHAIN:
            return LangchainChunker(config=self.config)

        if self.config.chunker_type == ChunkerType.LLM:
            raise ValueError("LLM chunker is deprecated. Choose other.")
        
        if self.config.chunker_type == ChunkerType.MATHEMATICAL:
            return MathematicalChunker(config=self.config)

        else:
            raise ChunkerNotImplementedError(self.config.chunker_type)


class ChunkerNotImplementedError(Exception):
    def __init__(self, chunker_type):
        msg = f"Chunker {chunker_type} is not implemented or not in chunker factory"
        
        super().__init__(msg)
