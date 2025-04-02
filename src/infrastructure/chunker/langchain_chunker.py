from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker


class LangchainChunker(Chunker):
    def chunk(book_path: str) -> list[Chunk]:
        raise NotImplementedError
