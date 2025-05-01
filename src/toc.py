from typing import List, Union
import os
from pydantic import BaseModel, Field
from llama_extract import LlamaExtract
from dotenv import load_dotenv
load_dotenv()


class Chapter(BaseModel):
    name: str = Field(description="Chapter name")
    start_page: int = Field(description="Start page number")
    
class TableOfContents(BaseModel):
    chapters: list[Chapter] = Field(description="List of chapters in the table of contents")

def main():
    
    extract = LlamaExtract(api_key=os.getenv("LLAMA_INDEX_KEY"))
    agent = extract.create_agent(name="toc-extractor", data_schema=TableOfContents)
    

    
if __name__ == "__main__":
    main()