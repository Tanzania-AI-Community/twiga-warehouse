from typing import Any

from together import Together

from src.config.settings import settings
from src.models.document import ParsedDocument

CLEAN_DOCUMENT_PROMPT = """
You are an expert at converting raw, unstructured data into a valid, desired format.
You will receive unstructured data from a book, and your task is to structure it so that it meets the criteria of the expected schema.
The content that you will receive is in Markdown, that part must go into "content". Mark content_type as Markdown too.
Important:
- Do not modify the contents of the data, your task is just to structure it!
- Output in a valid json format that meets the schema.
"""

class TogetherClient:
    def __init__(self):
        self.client = Together(api_key=settings.TOGETHER_AI_API_KEY)

    def clean_document(self, document: Any) -> ParsedDocument:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CLEAN_DOCUMENT_PROMPT},
            {"role": "user", "content": str(document)},
        ]
        
        response = self.client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            messages=messages,
            reasoning={"enabled": False},
            response_format={
                "type": "json_object",
                "schema": ParsedDocument.model_json_schema(),
            },
            temperature=0,
            max_tokens=100000,
        )
        return ParsedDocument.model_validate_json(response.choices[0].message.content)


client = TogetherClient()
