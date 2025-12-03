from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

from src.config.settings import settings
from src.domain.entities.table_of_contents import TableOfContents


toc_system_prompt = """
You are parsing the table of contents of a Tanzanian secondary education textbook.
Extract only real chapters; ignore glossary, appendix, references, acknowledgements, etc.
Each chapter must include `name`, `number`, and `start_page` fields that match the TableOfContents schema.
Use the page number where the chapter title first appears, and infer sequential numbering if the source omits digits.
"""
_structured_toc_llm = None


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
    structured_llm = _get_structured_toc_llm()

    messages = [
        SystemMessage(content=toc_system_prompt),
        HumanMessage(content=toc),
    ]

    toc_data = structured_llm.invoke(messages)
    return toc_data


def _get_structured_toc_llm():
    global _structured_toc_llm

    if _structured_toc_llm is not None:
        return _structured_toc_llm

    if not settings.GOOGLE_AI_API_KEY:
        raise ValueError("GOOGLE_AI_API_KEY is not configured.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=settings.GOOGLE_AI_API_KEY,
    )
    _structured_toc_llm = llm.with_structured_output(TableOfContents)
    return _structured_toc_llm
