import math
from pathlib import Path

import logging
from pydantic import BaseModel, Field
from pydantic_core._pydantic_core import ValidationError
from pypdf import PdfReader
from together import Together
from together.types import ChatCompletionResponse
from tqdm import tqdm

from src.application.mappers.llm_mapper import LLMMapper
from src.config.settings import settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, EmptyChunkerResponse
from src.domain.entities.table_of_contents import TableOfContents
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker
from src.domain.entities.table_of_contents import TableOfContents


logging.basicConfig(level=logging.INFO)
together = Together(api_key=settings.TOGETHER_AI_API_KEY)


class LLMChunk(BaseModel):
    chapter_number: int = Field(description="Chapter number of the chunk")
    page_number: int = Field(description="Page number of the chunk")
    content: str = Field(description="Parsed content of the chunk")


class LLMChapterBook(BaseModel):
    chunks: list[Chunk] = Field(
        description="List of the chunks that build the document. Order is the same as in the book"
    )


class LLMChunker(Chunker):
    def chunk(
        self,
        book_path: Path,
        table_of_contents: TableOfContents,
        text_initial_page: int = None,
    ) -> list[Chunk]:
        chunk_page_numbers: list[int] = []
        chunk_chapter_numbers: list[int] = []
        chunk_parsed_texts: list[str] = []

        current_batch = 1
        should_break_parsing = False

        for i, chapter in enumerate(table_of_contents.chapters):
            if should_break_parsing:
                break

            logging.info(f"Chunking chapter: {chapter.number}")

            start_page = chapter.start_page + text_initial_page - 1

            if i+1 <= len(table_of_contents.chapters) - 1: 
                end_page = table_of_contents.chapters[i+1].start_page + text_initial_page - 2
            else:
                end_page = self.config.last_page_number

            for batch_initial_page in range(start_page, end_page, self.config.page_batch_size):
                logging.debug(f"Parsing batch: {current_batch}")

                batch_end_page = min(batch_initial_page + self.config.page_batch_size + 1, end_page + 1)

                chapter_text = self._get_text_from_page_range(pdf_path=book_path, initial_page=batch_initial_page, end_page=batch_end_page)

                llm_response: ChatCompletionResponse = self.get_llm_response(text=chapter_text)

                try:
                    chapter_data = LLMChapterBook.model_validate_json(llm_response.choices[0].message.content)
                
                except ValidationError:
                    logging.error(f"LLM did not produce the expected JSON format in batch: {current_batch}.")

                    # should_break_parsing = True
                    # break
                    continue

                for llm_chunk in chapter_data.chunks:
                    chunk_parsed_texts.append(llm_chunk.content)
                    chunk_chapter_numbers.append(chapter.number)
                    chunk_page_numbers.append(batch_initial_page)

                current_batch += 1
 
        logging.info(f"Getting embeddings from {len(chunk_parsed_texts)} documents...\n")
        
        chunks: list[Chunk] = []
        embeddings = self.get_embeddings(chunk_parsed_texts)

        for page_number, chapter_number, parsed_text, embedding in zip(
            chunk_page_numbers,
            chunk_chapter_numbers,
            chunk_parsed_texts,
            embeddings
        ):
            chunk = LLMMapper.map(
                page_number=page_number,
                chapter_number=chapter_number,
                content_text=parsed_text,
                content_embedding=embedding,
                text_initial_page=text_initial_page,
            )
            chunks.append(chunk)
        
        if not chunks:
            raise EmptyChunkerResponse(book_path, self.config)

        return chunks

    def get_llm_response(self, text: str) -> ChatCompletionResponse:
        extract = together.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        """
                        The provided document is an elementary school text book from Tanzania. I want you to parse it without loosing
                        the important contents of the book. Bear in mind that there are some watermarks like "FOR ONLINE USE bla bla bla..."
                        that should not be taking into account for the parsing results.
                        
                        I want you to parse the text and output it in chunks. These chunks should be at paragraph level. The chunks
                        must be a json-like format, containing the chapter number it belongs to, the page number and obviously the text.
                        The text must be parsed and understandable to read.

                        Important: avoid focusing on figures, just focus on text: titles, subtitles, normal text...
                        """
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            model=self.config.llm_model_name,
            response_format={
                "type": "json_object",
                "schema": LLMChapterBook.model_json_schema(),
            },
        )

        return extract

    def get_embeddings(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        embeddings = []
        num_batches = math.ceil(len(texts) / batch_size)

        for i in tqdm(range(num_batches)):
            batch = texts[i * batch_size : (i + 1) * batch_size]

            response = together.embeddings.create(
                model=self.config.embedding_model_name,
                input=batch,
            )

            if not response.data:
                raise ValueError(f"Failed to generate embeddings for batch {i}")

            for j, embedding_data in enumerate(response.data):
                if embedding_data.embedding is None:
                    raise ValueError(f"Failed to generate embedding for text at index {i * batch_size + j}")
                embeddings.append(embedding_data.embedding)

        return embeddings

    @staticmethod
    def _get_text_from_page_range(pdf_path: Path, initial_page: int, end_page: int) -> str:
        reader = PdfReader(pdf_path)

        text = ""
        for page_num in range(initial_page, end_page):
            text += reader.pages[page_num-1].extract_text()
            
        return text
