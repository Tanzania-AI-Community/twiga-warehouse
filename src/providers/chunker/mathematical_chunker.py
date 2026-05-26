import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.chunk import TextChunk
from src.models.document import ParsedDocument
from src.models.toc import TableOfContents
from src.providers.chunker.base import Chunker, get_initial_chapter_page, get_page_chapter_number


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
    pattern=r"""
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
    flags=re.DOTALL | re.VERBOSE,
)
MATH_TAG_PATTERN = re.compile(pattern=r"</?math>")
IMAGE_REFERENCE_PATTERN = re.compile(pattern=r"!\[[^\]]*\]\([^)]+\)")


def _wrap_math_expressions(text: str) -> str:
    def replacer(match: re.Match) -> str:
        expression = match.group()
        if expression.startswith("<math>") and expression.endswith("</math>"):
            return expression
        if "<math>" in expression or "</math>" in expression:
            return expression
        return f"<math>{expression}</math>"

    return MATH_EXPRESSION_PATTERN.sub(repl=replacer, string=text)


def _update_math_balance(balance: int, text: str) -> int:
    for tag in MATH_TAG_PATTERN.findall(string=text):
        if tag == "<math>":
            balance += 1
        else:
            balance -= 1
    return balance


def _strip_image_references(text: str) -> str:
    return IMAGE_REFERENCE_PATTERN.sub(repl="", string=text)


def _merge_math_aware_chunks(chunks: list[TextChunk]) -> list[TextChunk]:
    merged_chunks: list[TextChunk] = []
    buffer_content = ""
    buffer_page_number = 0
    math_balance = 0

    for chunk in chunks:
        if buffer_page_number == 0:
            buffer_page_number = chunk.page_number

        buffer_content += chunk.content
        math_balance = _update_math_balance(balance=math_balance, text=chunk.content)

        if math_balance == 0:
            merged_chunks.append(
                TextChunk(
                    content=buffer_content,
                    page_number=buffer_page_number,
                    chapter_number=0,
                )
            )
            buffer_content = ""
            buffer_page_number = 0

    if buffer_content:
        merged_chunks.append(
            TextChunk(
                content=buffer_content,
                page_number=buffer_page_number,
                chapter_number=0,
            )
        )

    return merged_chunks


def _split_large_chunks(
    chunks: list[TextChunk],
    text_splitter: RecursiveCharacterTextSplitter,
    max_length: int,
) -> list[TextChunk]:
    size_safe_chunks: list[TextChunk] = []

    for chunk in chunks:
        if len(chunk.content) <= max_length:
            size_safe_chunks.append(chunk)
            continue

        for split_content in text_splitter.split_text(text=chunk.content):
            size_safe_chunks.append(
                TextChunk(
                    content=split_content,
                    page_number=chunk.page_number,
                    chapter_number=chunk.chapter_number,
                )
            )

    return size_safe_chunks


class MathematicalChunker(Chunker):
    MIN_LENGTH_TO_INCLUDE = 10
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 134
    UNWANTED_TEXT = [
        "FOR ONLINE USE ONLY",
        "DO NOT DUPLICATE",
        "PROPERTY OF THE UNITED REPUBLIC OF TANZANIA GOBVERNMENT",
        "Ministry of Education, Science and Technology",
        "For Online Use Only",
        "Student’s Book Form Two",
        "Geography for Secondary Schools",
    ]
    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_length_to_include: int = MIN_LENGTH_TO_INCLUDE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_length_to_include = min_length_to_include
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.UNWANTED_TEXT + self.DEFAULT_SEPARATORS,
            keep_separator=False,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(
        self,
        parsed_document: ParsedDocument,
        table_of_contents: TableOfContents,
        first_page_number: int,
        last_page_number: int | None = None,
    ) -> list[TextChunk]:
        raw_chunks: list[TextChunk] = []

        for page in parsed_document.pages:
            if last_page_number is not None and page.page_number > last_page_number:
                break

            cleaned_text = page.text
            for unwanted_text in self.UNWANTED_TEXT:
                cleaned_text = cleaned_text.replace(unwanted_text, "")
            cleaned_text = _strip_image_references(text=cleaned_text)
            cleaned_text = _wrap_math_expressions(text=cleaned_text)

            page_chunks = self.text_splitter.split_text(text=cleaned_text)
            for page_chunk in page_chunks:
                raw_chunks.append(
                    TextChunk(
                        content=page_chunk,
                        page_number=page.page_number,
                        chapter_number=0,
                    )
                )

        math_safe_chunks = _merge_math_aware_chunks(chunks=raw_chunks)
        size_safe_chunks = _split_large_chunks(
            chunks=math_safe_chunks,
            text_splitter=self.text_splitter,
            max_length=self.chunk_size,
        )
        initial_chapter_page = get_initial_chapter_page(
            table_of_contents=table_of_contents,
            first_page_number=first_page_number,
        )

        final_chunks: list[TextChunk] = []
        for chunk in size_safe_chunks:
            if chunk.page_number < initial_chapter_page - 1:
                continue
            if len(chunk.content) < self.min_length_to_include:
                continue

            chapter_number = get_page_chapter_number(
                page_number=chunk.page_number,
                first_page_number=first_page_number,
                table_of_contents=table_of_contents,
            )
            final_chunks.append(
                TextChunk(
                    content=chunk.content,
                    page_number=chunk.page_number - first_page_number + 1,
                    chapter_number=chapter_number,
                )
            )

        return final_chunks
