import json

from together import Together

from src.config.settings import settings
from src.models.toc import Chapter

CLEAN_PAGE_PROMPT = """
You are cleaning markdown extracted from a single textbook page.
You will receive markdown for one page only.
Important:
- Preserve the page content. Do not summarize or rewrite it.
- Return a valid JSON object with exactly one key: "content".
- Set "content" to the cleaned markdown string.
- Set "content" to null only if you cannot produce a valid cleaned page.
"""

PAGE_CLEAN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": ["string", "null"],
        }
    },
    "required": ["content"],
    "additionalProperties": False,
}


class TogetherClient:
    def __init__(self):
        self.client = Together(api_key=settings.TOGETHER_AI_API_KEY)

    def clean_page(
        self,
        markdown: str,
        chapter: Chapter | None = None,
    ) -> str | None:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_clean_page_prompt(chapter=chapter)},
            {"role": "user", "content": markdown},
        ]

        response = self.client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            messages=messages,
            reasoning={"enabled": False},
            response_format={
                "type": "json_object",
                "schema": PAGE_CLEAN_RESPONSE_SCHEMA,
            },
            temperature=0,
            max_tokens=100000,
            timeout=600,
        )
        response_content = response.choices[0].message.content
        if not isinstance(response_content, str):
            raise ValueError("Together clean_page response did not contain JSON text.")

        payload = json.loads(response_content)
        content = payload.get("content")
        if content is None:
            return None
        if not isinstance(content, str):
            raise ValueError("Together clean_page response field 'content' must be a string or null.")

        return content

    @staticmethod
    def _build_clean_page_prompt(chapter: Chapter | None) -> str:
        if chapter is None:
            return CLEAN_PAGE_PROMPT.strip()

        subchapter_lines = "\n".join(
            f"- {subchapter.name} (chapter page {subchapter.start_page})"
            for subchapter in chapter.subchapters
        )
        if not subchapter_lines:
            subchapter_lines = "- No subchapters were provided."

        return (
            f"{CLEAN_PAGE_PROMPT.strip()}\n\n"
            "This input is only one page from within the chapter, not the full chapter.\n"
            "Current chapter context from the table of contents:\n"
            f"- Chapter number: {chapter.number}\n"
            f"- Chapter name: {chapter.name}\n"
            "- Allowed `##` headings:\n"
            f"{subchapter_lines}\n\n"
            "Heading rules:\n"
            f"- Use `#` only for the chapter heading `Chapter {chapter.number}` / `{chapter.name}` when it is present on this page.\n"
            "- Use `##` only for subchapter headings that match the provided subchapter list and are present in the input.\n"
            "- If a heading is not the chapter heading or one of the listed subchapter headings, keep it at `###` or lower.\n"
            "- Do not invent missing chapter or subchapter headings.\n"
        )


client = TogetherClient()
