import logging

import requests


logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """Simple HTTP client for retrieving embeddings from an Ollama server."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _endpoint(self) -> str:
        return f"{self.base_url}/api/embed"

    def _request_embedding(self, prompt: str) -> list[float]:
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

    def embed_query(self, text: str) -> list[float]:
        return self._request_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from tqdm import tqdm

        l = []
        for text in tqdm(texts):
            try:
                l.append(self._request_embedding(text))
            except:
                l.append([])

        return l


def get_embedding_client(model_name: str | None = None, base_url: str | None = None) -> OllamaEmbeddingClient:
    resolved_model = model_name or "mxbai-embed-large"
    resolved_url = (base_url or "http://localhost:11434/").rstrip("/")
    return OllamaEmbeddingClient(base_url=resolved_url, model=resolved_model)


def get_embeddings(
    texts: list[str],
    model_name: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    client = get_embedding_client(model_name=model_name, base_url=base_url)
    return client.embed_documents(texts)
