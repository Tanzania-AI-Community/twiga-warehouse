from pathlib import Path

from langchain_core.documents import Document
import mistralai

from src.config.settings import settings


class MistralParser():
    def __init__(self, book_path: Path):
        self.client = mistralai.Mistral(api_key=settings.MISTRAL_API_KEY)
        self.book_path = Path(book_path)

    def load(self) -> list[Document]:
        uploaded_pdf = self.client.files.upload(
            file={
                "file_name": self.book_path.name,
                "content": self.book_path.open("rb"),
            },
            purpose="ocr",
        )

        signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)

        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=True
        )

        return self.response_to_documents(ocr_response=ocr_response)

    @staticmethod
    def response_to_documents(
        ocr_response: mistralai.models.ocrresponse.OCRResponse
    ) -> list[Document]:    
        docs = []
        for i, page in enumerate(ocr_response.pages, start=1):
            docs.append(
                Document(
                    page_content=page.markdown + "\n",
                    metadata={
                        "page_number": i,
                        "page_label": i,
                    }
                )
            )

        return docs
