from pydantic import BaseModel, Field
from pypdf import PdfReader
from together import Together
import os

from dotenv import load_dotenv
load_dotenv()

together = Together(api_key=os.getenv("TOGETHER_API_KEY"))

class Chapter(BaseModel):
    name: str = Field(description="Chapter name")
    number: int = Field(description="Chapter number")
    start_page: int = Field(description="Chapter start page number")
    
class TableOfContents(BaseModel):
    chapters: list[Chapter] = Field(description="List of chapters in the table of contents")


def get_raw_page_text(pdf_path: str, toc_page_number: int | list[int]):
    """ Extracts text from a PDF file given a page number or list of page numbers.

    Args:
        pdf_path (str): Path to the PDF file.
        toc_page_number (int | list[int]): Page number or list of page numbers.

    Returns:
        str: Extracted text from the specified page(s).
    """

    reader = PdfReader(pdf_path)

    if isinstance(toc_page_number, int):
        toc_page_number = [toc_page_number]
    
    text = ""
    for page_num in toc_page_number:
        text += reader.pages[page_num-1].extract_text()
        
    return text


def get_toc(pdf_path: str, toc_page_number: int | list[int]):
    """ Extracts chapter information from table of contents

    Args:
        pdf_path (str): Path to the PDF file.
        toc_page_number (int | list[int]): Page number or list of page numbers.

    Returns:
        TableOfContents: Extracted chapter information.
    """

    toc = get_raw_page_text(pdf_path, toc_page_number)

    extract = together.chat.completions.create(
        messages=[
            {
                "role": "system",
                        "content": "Extract the chapter names, chapter numbers and chapter page number from this table of contents. Note: ONLY extract the CHAPTERS! DO NOT extract the glossary, appendix, references, etc. ",
            },
            {
                "role": "user",
                "content": toc,
            },
        ],
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        response_format={
            "type": "json_object",
            "schema": TableOfContents.model_json_schema(),
        },
    )

    toc_data = TableOfContents.model_validate_json(extract.choices[0].message.content)
    return toc_data