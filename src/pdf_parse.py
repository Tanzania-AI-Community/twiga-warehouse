import json
import os
from typing import List
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
unstructured_api_url=os.getenv("UNSTRUCTURED_API_URL")

def create_documents():
    """Create documents from a PDF file and save them as JSON files.

    Args:
        file_name (str): The name of the PDF file in the data/raw directory
    """
    
    file_path = r"C:\Users\ADMIN\Desktop\KTHAIS\twiga-warehouse\data\geo_form2.pdf"

    loader = UnstructuredLoader(
        file_path=file_path,
        strategy="hi_res",
        unique_element_ids=True,
        
        partition_via_api=True,
        coordinates=True,
        api_key=unstructured_api_key,
        url=unstructured_api_url,
    )

    docs: List[Document] = []
    for doc in loader.lazy_load():
        docs.append(doc)

    output_path = r"C:\Users\ADMIN\Desktop\KTHAIS\twiga-warehouse\data\geo_form2.md"
    with open(output_path, "w", encoding="utf-8") as f:
        docs_dict = [doc.model_dump() for doc in docs]
        json.dump(docs_dict, f, ensure_ascii=False, indent=2)

    print("Document generated.")
    
    return docs

create_documents()