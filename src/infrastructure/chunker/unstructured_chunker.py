from dotenv import load_dotenv
import os
import math

from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from together import Together

from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, EmptyChunkerResponse
from src.domain.mappers.unstructured_mapper import UnstructuredMapper


class UnstructuredChunker(Chunker):
    def chunk(self, book_path: str) -> list[Chunk]:
        api_key, api_url = self._get_keys()

        loader = UnstructuredLoader(
            file_path=book_path,
            strategy="hi_res",
            unique_element_ids=True,
            partition_via_api=True,
            coordinates=True,
            api_key=api_key,
            url=api_url,
        )

        documents: list[Document] = []
        parsed_text: list[str] = []
        for doc in loader.lazy_load():
            if len(doc.page_content) == 0 or doc.metadata["category"] != "NarrativeText":  # TODO: recheck this again
                continue

            documents.append(doc)
            parsed_text.append(doc.page_content)

        chunks: list[Chunk] = []        
        embeddings = self.get_embeddings(parsed_text)
        for doc, embedding in zip(documents, embeddings):
            chunk = UnstructuredMapper.map(document=doc, content_embedding=embedding)

            chunks.append(chunk)
        
        if not chunks:
            raise EmptyChunkerResponse(book_path)

        return chunks

    @staticmethod
    def _get_keys() -> tuple[str, str]:
        # TODO: refactor
        load_dotenv()

        unstructured_api_key =  os.getenv("UNSTRUCTURED_API_KEY")
        unstructured_api_url = os.getenv("UNSTRUCTURED_API_URL")

        return unstructured_api_key, unstructured_api_url

    @staticmethod
    def get_embeddings(texts: list[str], batch_size: int = 16) -> list[list[float]]:
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

        embeddings = []
        num_batches = math.ceil(len(texts) / batch_size)

        for i in range(num_batches):
            batch = texts[i * batch_size : (i + 1) * batch_size]

            response = client.embeddings.create(
                model="BAAI/bge-large-en-v1.5",
                input=batch,
            )

            if not response.data:
                raise ValueError(f"Failed to generate embeddings for batch {i}")

            for j, embedding_data in enumerate(response.data):
                if embedding_data.embedding is None:
                    raise ValueError(f"Failed to generate embedding for text at index {i * batch_size + j}")
                embeddings.append(embedding_data.embedding)

        return embeddings
