import math
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from tqdm import tqdm

from src.application.mappers.langchain_mapper import LangchainMapper
from src.config.settings import settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, EmptyChunkerResponse
from src.domain.entities.table_of_contents import TableOfContents
from src.infrastructure.embedder.ollama_embedder import get_embeddings


logging.basicConfig(level=logging.INFO)


class LangchainChunker(Chunker):
    def chunk(
        self,
        book_path: Path,
        table_of_contents: TableOfContents,
        text_initial_page: int = None,
        min_length_to_be_included: int = 10,
    ) -> list[Chunk]:
        loader = PyPDFLoader(book_path)
        docs = loader.load()
        
        page_of_initial_chapter = (
            table_of_contents.chapters[0].start_page + text_initial_page
            if table_of_contents.chapters and text_initial_page
            else 0
        )

        separators = [
            "FOR ONLINE USE ONLY",
            "DO NOT DUPLICATE",
            "PROPERTY OF THE UNITED REPUBLIC OF TANZANIA GOBVERNMENT",
            "Ministry of Education, Science and Technology",
            "For Online Use Only",
            "Student’s Book Form Two",
            "Geography for Secondary Schools",
        ]

        default_separators = [
            "\n\n",
            "\n",
            " ",
            "",
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators + default_separators,
            keep_separator=False,
            chunk_size=250,
            chunk_overlap=30,
        )

        initial_documents: list[Document] = text_splitter.split_documents(docs)

        documents: list[Document] = []
        document_chapters: list[int] = []
        parsed_text: list[str] = []

        for doc in initial_documents:
            doc_page = int(doc.metadata["page_label"])
            if doc_page < page_of_initial_chapter - 1:
                continue

            page_content: str = doc.page_content

            for unwanted_text in separators:  # just in case
                page_content = page_content.replace(unwanted_text, "")

            if len(page_content) < min_length_to_be_included:
                continue

            doc.page_content = page_content
            doc_chapter = self.get_document_chapter(
                doc_page=doc_page,
                text_initial_page=text_initial_page,
                table_of_contents=table_of_contents,
            )

            documents.append(doc)
            document_chapters.append(doc_chapter)
            parsed_text.append(page_content)

        chunks: list[Chunk] = []        

        print(f"Getting embeddings from {len(documents)} documents...\n")

        embeddings = get_embeddings(parsed_text, model_name=self.config.embedding_model_name)
        for doc, chapter_number, embedding in zip(documents, document_chapters, embeddings):
            chunk = LangchainMapper.map(
                document=doc,
                content_embedding=embedding,
                chapter_number=chapter_number,
                text_initial_page=text_initial_page,
            )
            chunks.append(chunk)
        
        if not chunks:
            raise EmptyChunkerResponse(book_path, self.config)

        return chunks

    @staticmethod
    def get_document_chapter(doc_page: int, text_initial_page: int, table_of_contents: TableOfContents) -> int:
        doc_chapter = 0
        for chapter in table_of_contents.chapters:
            if doc_page < chapter.start_page + text_initial_page - 1:
                break
            
            doc_chapter = chapter.number
        
        return doc_chapter
