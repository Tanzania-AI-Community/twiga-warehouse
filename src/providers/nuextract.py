import base64
import json
from pathlib import Path

import fitz
from openai import OpenAI

from src.config.settings import settings


class NuExtractClient:
    MODEL_NAME = "numind/NuExtract3"

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str = MODEL_NAME,
        dpi: int = 100,
        temperature: float = 0.2,
    ):
        resolved_base_url = self.resolve_base_url(base_url=base_url)
        if not resolved_base_url:
            raise ValueError("CUSTOM_OCR_MODEL_URL is required to use the hosted OCR model.")

        self.client = OpenAI(
            api_key=settings.CUSTOM_OCR_MODEL_API_KEY,
            base_url=resolved_base_url,
        )
        self.base_url = resolved_base_url
        self.model_name = model_name
        self.dpi = dpi
        self.temperature = temperature

    @staticmethod
    def resolve_base_url(base_url: str | None) -> str:
        resolved_base_url = (base_url or settings.CUSTOM_OCR_MODEL_URL or "").strip().rstrip("/")
        if not resolved_base_url:
            return ""
        if resolved_base_url.endswith("/v1"):
            return resolved_base_url
        return f"{resolved_base_url}/v1"

    def render_pdf_pages_to_data_urls(
        self,
        *,
        pdf_path: Path,
        page_numbers: list[int],
    ) -> list[str]:
        with fitz.open(pdf_path) as document:
            return self.render_document_pages_to_data_urls(
                document=document,
                page_numbers=page_numbers,
            )

    def render_document_pages_to_data_urls(
        self,
        *,
        document: fitz.Document,
        page_numbers: list[int],
    ) -> list[str]:
        data_urls: list[str] = []

        for page_number in page_numbers:
            if page_number < 1 or page_number > document.page_count:
                raise ValueError(
                    f"Requested page {page_number} is outside the document range 1-{document.page_count}."
                )

            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
            png_base64 = base64.b64encode(pixmap.tobytes("png")).decode("utf-8")
            data_urls.append(f"data:image/png;base64,{png_base64}")

        return data_urls

    def extract_structured(
        self,
        *,
        page_data_urls: list[str],
        template: dict[str, object],
        instructions: str | None = None,
        temperature: float | None = None,
    ) -> object:
        if not page_data_urls:
            raise ValueError("NuExtract received an empty image batch.")

        user_content: list[dict[str, object]] = []
        if instructions:
            user_content.append(
                {
                    "type": "text",
                    "text": instructions,
                }
            )

        user_content.extend(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
            for data_url in page_data_urls
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            extra_body={
                "chat_template_kwargs": {
                    "template": json.dumps(template, indent=4),
                    "enable_thinking": False,
                }
            },
        )
        response_body = self.extract_completion_text(response.choices[0].message.content)
        if not response_body:
            raise ValueError("NuExtract response did not contain any text.")

        return self.parse_response_payload(response_body=response_body)

    @staticmethod
    def parse_response_payload(response_body: str) -> object:
        candidates = [response_body.strip()]
        fenced_candidate = NuExtractClient.strip_code_fences(response_body=response_body)
        if fenced_candidate != candidates[0]:
            candidates.append(fenced_candidate)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return {"text": response_body}

    @staticmethod
    def extract_completion_text(message_content: object) -> str:
        if isinstance(message_content, str):
            return message_content

        if isinstance(message_content, list):
            text_parts: list[str] = []

            for item in message_content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

            return "\n".join(text_parts)

        return ""

    @staticmethod
    def strip_code_fences(response_body: str) -> str:
        stripped = response_body.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            return stripped

        return "\n".join(lines[1:-1]).strip()
