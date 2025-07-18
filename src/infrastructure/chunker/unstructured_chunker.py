from dotenv import load_dotenv
import os
import math

from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from together import Together

from src.application.mappers.unstructured_mapper import UnstructuredMapper
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, EmptyChunkerResponse
from src.domain.entities.table_of_contents import TableOfContents

from pypdf import PdfReader
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


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


    def _load_pages(self, book_path: str, start_page: int, end_page: int) -> UnstructuredLoader:

        api_key, api_url = self._get_keys()

        loader = UnstructuredLoader(
            file_path=book_path,
            strategy="hi_res",
            unique_element_ids=True,
            partition_via_api=True,
            coordinates=True,
            api_key=api_key,
            url=api_url,
            split_pdf_page=True,
            split_pdf_page_range=[start_page, end_page]
        )

        return loader


    def _get_page_count(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            return len(reader.pages)
        

    def chunk(self, book_path: str, toc: TableOfContents, text_initial_page: int = None) -> list[Chunk]:
        documents: list[Document] = []
        parsed_text: list[str] = []
        chunks: list[Chunk] = []  

        page_count = self._get_page_count(book_path)

        logging.info

        logging.info(f"Page count: {page_count}. Begin chunking by chapter...")
        
        for i, chapter in enumerate(tqdm(toc.chapters)):

            last_page = (toc.chapters[i+1].start_page - 1) if i+1 < len(toc.chapters) else page_count
            loader = self._load_pages(book_path, chapter.start_page, last_page)

            logging.info(f"Chunking chapter {chapter.number}...")

            for doc in loader.lazy_load():
                if len(doc.page_content) == 0 or doc.metadata["category"] != "NarrativeText":  # TODO: recheck this again
                    continue

                documents.append(doc)
                parsed_text.append(doc.page_content)
  
            embeddings = self.get_embeddings(parsed_text)
            for doc, embedding in zip(documents, embeddings):
                chunk = UnstructuredMapper.map(document=doc, content_embedding=embedding, chapter_number=chapter.number)
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
