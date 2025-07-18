from pypdf import PdfReader
from together import Together

from src.config.settings import settings
from src.domain.entities.table_of_contents import TableOfContents


together = Together(api_key=settings.TOGETHER_AI_API_KEY)
llm_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


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


def get_table_of_contents(pdf_path: str, toc_page_number: int | list[int]):
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
                "content": (
                    """
                    Extract the chapter names, chapter numbers and chapter page number from this table of contents.
                    Note: ONLY extract the CHAPTERS! DO NOT extract the glossary, appendix, references, etc.
                    Also, the chapter page number must be the one where the title is located!
                    Obviously, the first chapter will always have 1 as the start page.
                    """
                ),
            },
            {
                "role": "user",
                "content": toc,
            },
        ],
        model=llm_model,
        response_format={
            "type": "json_object",
            "schema": TableOfContents.model_json_schema(),
        },
    )

    toc_data = TableOfContents.model_validate_json(extract.choices[0].message.content)
    return toc_data
