import re
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

from src.config.settings import settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.chunker import Chunker, EmptyChunkerResponse
from src.domain.entities.table_of_contents import TableOfContents
from src.infrastructure.embedder.embedding_router import get_embeddings
from src.infrastructure.parser.firecrawl_parser import FirecrawlParser

logging.basicConfig(level=logging.INFO)

class LangchainChunker(Chunker):
    def chunk(
        self,
        book_path: Path,
        table_of_contents: TableOfContents,
        text_initial_page: int = None,
        min_length_to_be_included: int = 100,
    ) -> list[Chunk]:
        
        parser = FirecrawlParser(book_path=book_path)
        docs = parser.load() 

        full_markdown_text = ""
        for doc in docs:
            clean_md = self.clean_textbook_content(doc.page_content)            
            page_num = doc.metadata.get("page_number")
            if not str(page_num).isdigit():
                page_num = 0
            

            full_markdown_text += f"\n\n\n\n{clean_md}"
            
        headers_to_split_on = [
            ("#", "Header_1"),       
            ("##", "Header_2"),      
            ("###", "Header_3"),     
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False 
        )
        md_header_splits = markdown_splitter.split_text(full_markdown_text)

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=150
        )
        all_splits = char_splitter.split_documents(md_header_splits)
        
        current_page = docs[0].metadata.get("page_number", 1) if docs else 1
        if not str(current_page).isdigit():
            current_page = 1


        clean_texts = []
        page_numbers = []
        chapter_numbers = []
        metadatas = []


        for split in all_splits:

            page_matches = re.findall(r"", split.page_content)
            valid_pages = [p for p in page_matches if p.strip().isdigit()]
            
            if valid_pages:
                current_page = int(valid_pages[-1])
                
            clean_content = re.sub(r"\n*", "", split.page_content).strip()
            
            if len(clean_content) >= min_length_to_be_included:
                doc_chapter = self.get_document_chapter(current_page, text_initial_page or 0, table_of_contents)
                
                # Append to our parallel lists
                clean_texts.append(clean_content)
                page_numbers.append(current_page)
                chapter_numbers.append(doc_chapter)
                

                split.metadata["page_number"] = current_page
                metadatas.append(split.metadata)

        
        embeddings = get_embeddings(
            clean_texts,
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model_name,
        )

        final_chunks = []
        
        for i, (text, page_num, chapter_num, meta, emb) in enumerate(zip(clean_texts, page_numbers, chapter_numbers, metadatas, embeddings)):
            chunk = Chunk(
                chunk_id=i, 
                content=text,
                metadata=meta,
                chapter_number=chapter_num,  
                page_number=page_num,        
                embedding=emb                
            )
            final_chunks.append(chunk)
            
        if not final_chunks:
            raise EmptyChunkerResponse(book_path, self.config)

        return final_chunks

    @staticmethod
    def get_document_chapter(doc_page: int, text_initial_page: int, table_of_contents: TableOfContents) -> int:
        doc_chapter = 0
        for chapter in table_of_contents.chapters:
            if doc_page < chapter.start_page + text_initial_page - 1:
                break
            doc_chapter = chapter.number
        return doc_chapter
    
    @staticmethod
    def clean_textbook_content(text: str) -> str:
        watermarks = [
            r"(?i)property of the united republic of tanzania government",
            r"PROPER Ot hte ONT Ee REE ODL Ot TANZANIA OOD VERN",
            r"(?i)for online use only",
            r"INE USE ONLY",
            r"(?i)do not duplicate",
            r"OT DUPLICATE",
            r"©.*?Se,?",       
            r"ne \| ee"        
        ]
        
        for wm in watermarks:
            text = re.sub(wm, "", text)
            
        text = re.sub(r"(?i)source:\s*https?://\S+", "", text)
        text = re.sub(r" {2,}", " ", text)       
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()