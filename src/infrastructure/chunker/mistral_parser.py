# parse book with mistral -> each page as a document -> chunk with the original chunker

import os
from dotenv import load_dotenv
from mistralai import Mistral
import mistralai
from langchain_core.documents import Document
import pathlib
from typing import List

class MistralParse():

    def __init__(self, book_path: str):
        
        load_dotenv()
        self.api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=self.api_key)
        self.book_path = book_path

    def load(self) -> List[Document]:

        uploaded_pdf = self.client.files.upload(
        file={
            "file_name": self.book_path,
            "content": open(self.book_path, "rb"),
        },
        purpose="ocr"
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

        return self.parsed_to_document(ocr_response=ocr_response)

    @staticmethod
    def parsed_to_document(ocr_response: mistralai.models.ocrresponse.OCRResponse) -> List[Document]:
        
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


