import os
import pickle
import re
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.application.mappers.langchain_mapper import LangchainMapper
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, EmptyChunkerResponse
from src.domain.entities.table_of_contents import TableOfContents
from src.infrastructure.parser.mistral_parser import MistralParser
from src.infrastructure.embedder.ollama_embedder import get_embeddings

logging.basicConfig(level=logging.INFO)


MATH_ENVIRONMENT_NAMES = (
    "align",
    "align*",
    "equation",
    "equation*",
    "gather",
    "gather*",
    "multline",
    "multline*",
)

MATH_EXPRESSION_PATTERN = re.compile(
    r"""
    (\\begin\{(?P<env>""" + "|".join(MATH_ENVIRONMENT_NAMES) + r""")\}.*?\\end\{(?P=env)\})
    |
    (\$\$.*?\$\$)
    |
    (\\\[.*?\\\])
    |
    (\\\(.*?\\\))
    |
    (\$(?:\\.|[^$\\])+\$)
    """,
    re.DOTALL | re.VERBOSE,
)

MATH_TAG_PATTERN = re.compile(r"</?math>")

IMAGE_REFERENCE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def _wrap_math_expressions(text: str) -> str:
    def replacer(match: re.Match) -> str:
        expression = match.group(0)
        if expression.startswith("<math>") and expression.endswith("</math>"):
            return expression
        if "<math>" in expression or "</math>" in expression:
            return expression
        return f"<math>{expression}</math>"

    return MATH_EXPRESSION_PATTERN.sub(replacer, text)


def _update_math_balance(balance: int, text: str) -> int:
    for tag in MATH_TAG_PATTERN.findall(text):
        if tag == "<math>":
            balance += 1
        else:
            balance -= 1
    return balance


def _merge_math_aware_documents(documents: list[Document]) -> list[Document]:
    merged_documents: list[Document] = []
    buffer_text: str = ""
    buffer_metadata: dict | None = None
    math_balance = 0

    for doc in documents:
        if buffer_metadata is None:
            buffer_metadata = {**doc.metadata}

        buffer_text += doc.page_content
        math_balance = _update_math_balance(math_balance, doc.page_content)

        if math_balance == 0:
            merged_documents.append(Document(page_content=buffer_text, metadata=buffer_metadata))
            buffer_text = ""
            buffer_metadata = None

    if buffer_text:
        merged_documents.append(Document(page_content=buffer_text, metadata=buffer_metadata or {}))

    return merged_documents


def _strip_image_references(text: str) -> str:
    return IMAGE_REFERENCE_PATTERN.sub("", text)


class MathematicalChunker(Chunker):
    MIN_LENGTH_TO_BE_INCLUDED = 10
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 134

    def chunk(
        self,
        book_path: Path,
        table_of_contents: TableOfContents,
        text_initial_page: int = None,
    ) -> list[Chunk]:
        loader = MistralParser(book_path)
        docs = loader.load()

        page_of_initial_chapter = (
            table_of_contents.chapters[0].start_page + text_initial_page
            if table_of_contents and len(table_of_contents.chapters) > 0 and text_initial_page
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

        processed_docs: list[Document] = []

        for doc in docs:
            cleaned_content = doc.page_content
            for unwanted_text in separators:
                cleaned_content = cleaned_content.replace(unwanted_text, "")
            cleaned_content = _strip_image_references(cleaned_content)
            cleaned_content = _wrap_math_expressions(cleaned_content)

            processed_docs.append(
                Document(
                    page_content=cleaned_content,
                    metadata={**doc.metadata},
                )
            )

        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators + default_separators,
            keep_separator=False,
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )

        initial_documents: list[Document] = text_splitter.split_documents(processed_docs)
        math_safe_documents: list[Document] = _merge_math_aware_documents(initial_documents)

        documents: list[Document] = []
        document_chapters: list[int] = []
        parsed_text: list[str] = []

        for doc in math_safe_documents:
            doc_page = int(doc.metadata["page_label"])
            if doc_page < page_of_initial_chapter - 1:
                continue

            page_content: str = doc.page_content

            if len(page_content) < self.MIN_LENGTH_TO_BE_INCLUDED:
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

        embeddings = get_embeddings(texts=parsed_text, model_name=self.config.embedding_model_name)
        for doc, chapter_number, embedding in zip(documents, document_chapters, embeddings):
            if len(embedding) == 0:
                continue

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
