import logging
from typing import List

import requests
from langchain_openai import OpenAIEmbeddings
from langchain_together.embeddings import TogetherEmbeddings
from pydantic import SecretStr



logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """Simple HTTP client for retrieving embeddings from an Ollama server."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _endpoint(self) -> str:
        return f"{self.base_url}/api/embed"

    def _request_embedding(self, prompt: str) -> List[float]:
        payload = {"model": self.model, "input": prompt}

        try:
            response = requests.post(
                self._endpoint(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.error("Ollama embedding request failed", exc_info=True)
            raise RuntimeError(f"Failed to fetch embedding from Ollama: {exc}")

        data = response.json()
        embedding = data.get("embeddings")
        if embedding is None:
            raise ValueError("Ollama response did not include an 'embedding' field")
        return embedding[0]

    def embed_query(self, text: str) -> List[float]:
        return self._request_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._request_embedding(text) for text in texts]


def get_embedding_client():
    model_name = "mxbai-embed-large"
    base_url = "https://b6574b1e8fb6.ngrok-free.app/"

    return OllamaEmbeddingClient(base_url=base_url, model=model_name)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts."""
    client = get_embedding_client()
    return client.embed_documents(texts)
