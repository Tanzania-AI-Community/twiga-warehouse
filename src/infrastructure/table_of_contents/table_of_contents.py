import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from pydantic import SecretStr
from together import Together

from src.config.settings import settings
from src.domain.entities.table_of_contents import (
    TableOfContents,
    TableOfContentsParserConfig,
    TableOfContentsParserType,
)


toc_system_prompt = """
You are parsing the table of contents of a Tanzanian secondary education textbook.
Extract only real chapters; ignore glossary, appendix, references, acknowledgements, etc.
Each chapter must include `name`, `number`, and `start_page` fields that match the TableOfContents schema.
Use the page number where the chapter title first appears, and infer sequential numbering if the source omits digits.
"""


def get_raw_page_text(pdf_path: Path, toc_page_number: int | list[int]):
    """ Extracts text from a PDF file given a page number or list of page numbers.

    Args:
        pdf_path (Path): Path to the PDF file.
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


def get_table_of_contents(
    pdf_path: Path,
    toc_page_number: int | list[int],
    parser_config: TableOfContentsParserConfig,
):
    """ Extracts chapter information from table of contents

    Args:
        pdf_path (Path): Path to the PDF file.
        toc_page_number (int | list[int]): Page number or list of page numbers.
        parser_config (TableOfContentsParserConfig): Configuration for TOC parsing.

    Returns:
        TableOfContents: Extracted chapter information.
    """

    if parser_config.parser_type == TableOfContentsParserType.NONE:
        logging.warning("Table of contents parsing disabled; no table of contents will be obtained from the book.")
        return TableOfContents(chapters=[])

    toc = get_raw_page_text(pdf_path, toc_page_number)
    messages = [
        SystemMessage(content=toc_system_prompt),
        HumanMessage(content=toc),
    ]

    structured_llm = _select_structured_toc_llm(parser_config)

    return structured_llm.invoke(messages)


def _select_structured_toc_llm(parser_config: TableOfContentsParserConfig):
    if parser_config.parser_type == TableOfContentsParserType.TOGETHER:
        return _get_structured_toc_llm_together()
    if parser_config.parser_type == TableOfContentsParserType.OLLAMA:
        return _get_structured_toc_llm_ollama(parser_config)
    return _get_structured_toc_llm_gemini()


def _get_structured_toc_llm_gemini():
    if not settings.GOOGLE_AI_API_KEY:
        raise ValueError("GOOGLE_AI_API_KEY is not configured.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=settings.GOOGLE_AI_API_KEY,
    )
    return llm.with_structured_output(TableOfContents)


def _get_structured_toc_llm_together():
    if not settings.TOGETHER_AI_API_KEY:
        raise ValueError("TOGETHER_AI_API_KEY is not configured.")

    class _TogetherStructuredTOC:
        def __init__(self):
            self.client = Together(api_key=settings.TOGETHER_AI_API_KEY)

        def invoke(self, messages: list):
            formatted_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    role = "system"
                elif isinstance(message, HumanMessage):
                    role = "user"
                else:
                    continue

                formatted_messages.append({"role": role, "content": message.content})

            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                messages=formatted_messages,
                response_format={
                    "type": "json_object",
                    "schema": TableOfContents.model_json_schema(),
                },
                temperature=0,
            )

            return TableOfContents.model_validate_json(response.choices[0].message.content)

    return _TogetherStructuredTOC()


def _get_structured_toc_llm_ollama(parser_config: TableOfContentsParserConfig):
    model_name = parser_config.ollama_model_name or "llama3.2"
    base_url = parser_config.ollama_base_url or "http://localhost:11434/v1"

    api_key = (
        parser_config.api_key.get_secret_value()
        if parser_config.api_key
        else "ollama"
    )

    llm = ChatOpenAI(
        api_key=SecretStr(api_key),
        model=model_name,
        base_url=base_url,
        temperature=0,
    )

    return llm.with_structured_output(TableOfContents)
