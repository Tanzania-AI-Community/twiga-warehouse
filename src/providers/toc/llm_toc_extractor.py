import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from pydantic import SecretStr
from together import Together

from src.config.settings import settings
from src.models.toc import TableOfContents, TableOfContentsParserConfig, TableOfContentsParserType


TOC_SYSTEM_PROMPT = """
You are parsing the table of contents of a Tanzanian secondary education textbook.
Extract only real chapters; ignore glossary, appendix, references, acknowledgements, etc.
Each chapter must include `name`, `number`, and `start_page` fields that match the TableOfContents schema.
Use the page number where the chapter title first appears, and infer sequential numbering if the source omits digits.
"""


def extract_table_of_contents(
    pdf_path: Path,
    toc_page_numbers: int | list[int],
    parser_config: TableOfContentsParserConfig,
) -> TableOfContents:
    if parser_config.parser_type == TableOfContentsParserType.NONE:
        logging.warning("Table of contents parsing disabled.")
        return TableOfContents(chapters=[])

    toc_text = get_raw_page_text(
        pdf_path=pdf_path,
        toc_page_numbers=toc_page_numbers,
    )
    messages = [
        SystemMessage(content=TOC_SYSTEM_PROMPT),
        HumanMessage(content=toc_text),
    ]
    llm = get_structured_toc_llm(parser_config=parser_config)
    return llm.invoke(messages)


def get_raw_page_text(
    pdf_path: Path,
    toc_page_numbers: int | list[int],
) -> str:
    reader = PdfReader(stream=pdf_path)
    page_numbers = toc_page_numbers if isinstance(toc_page_numbers, list) else [toc_page_numbers]

    raw_text = ""
    for page_number in page_numbers:
        raw_text += reader.pages[page_number - 1].extract_text() or ""

    return raw_text


def get_structured_toc_llm(
    parser_config: TableOfContentsParserConfig,
):
    if parser_config.parser_type == TableOfContentsParserType.TOGETHER:
        return get_together_toc_llm()

    if parser_config.parser_type == TableOfContentsParserType.OLLAMA:
        return get_ollama_toc_llm(parser_config=parser_config)

    return get_gemini_toc_llm()


def get_gemini_toc_llm():
    if not settings.GOOGLE_AI_API_KEY:
        raise ValueError("GOOGLE_AI_API_KEY is not configured.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=settings.GOOGLE_AI_API_KEY,
    )
    return llm.with_structured_output(TableOfContents)


def get_together_toc_llm():
    if not settings.TOGETHER_AI_API_KEY:
        raise ValueError("TOGETHER_AI_API_KEY is not configured.")

    class TogetherTocClient:
        def __init__(self):
            self.client = Together(api_key=settings.TOGETHER_AI_API_KEY)

        def invoke(self, messages: list[SystemMessage | HumanMessage]) -> TableOfContents:
            formatted_messages: list[dict[str, str]] = []

            for message in messages:
                role = "system" if isinstance(message, SystemMessage) else "user"
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

    return TogetherTocClient()


def get_ollama_toc_llm(
    parser_config: TableOfContentsParserConfig,
):
    model_name = parser_config.ollama_model_name or "llama3.2"
    base_url = parser_config.ollama_base_url or "http://localhost:11434/v1"
    api_key = parser_config.api_key.get_secret_value() if parser_config.api_key else "ollama"

    llm = ChatOpenAI(
        api_key=SecretStr(api_key),
        model=model_name,
        base_url=base_url,
        temperature=0,
    )
    return llm.with_structured_output(TableOfContents)
