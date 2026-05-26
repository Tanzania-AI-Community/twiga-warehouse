from pathlib import Path

import mistralai

from src.config.settings import settings
from src.models.document import ParsedDocument, ParsedPage


class MistralOcrParser:
    def __init__(self, api_key: str | None = None, model_name: str = "mistral-ocr-latest"):
        resolved_api_key = api_key or settings.MISTRAL_API_KEY
        if not resolved_api_key:
            raise ValueError("MISTRAL_API_KEY is required to use Mistral OCR parser.")

        self.client = mistralai.Mistral(api_key=resolved_api_key)
        self.model_name = model_name

    def parse(self, pdf_path: Path) -> ParsedDocument:
        normalized_path = Path(pdf_path)

        with normalized_path.open(mode="rb") as handle:
            uploaded_pdf = self.client.files.upload(
                file={
                    "file_name": normalized_path.name,
                    "content": handle,
                },
                purpose="ocr",
            )

        signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)

        ocr_response = self.client.ocr.process(
            model=self.model_name,
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=True,
        )

        return self.response_to_parsed_document(ocr_response=ocr_response)

    @staticmethod
    def response_to_parsed_document(
        ocr_response: mistralai.models.ocrresponse.OCRResponse,
    ) -> ParsedDocument:
        pages: list[ParsedPage] = []

        for page_index, page in enumerate(ocr_response.pages, start=1):
            pages.append(ParsedPage(page_number=page_index, text=f"{page.markdown}\n"))

        return ParsedDocument(pages=pages)
