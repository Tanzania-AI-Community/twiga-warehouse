import logging

import requests


logger = logging.getLogger(__name__)


EMBEDDING_MODELS = {
    "multilingual-large": "intfloat/multilingual-e5-large-instruct",
}
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS["multilingual-large"]


def resolve_embedding_model_name(model_name: str | None) -> str:
    if model_name is None:
        return DEFAULT_EMBEDDING_MODEL
    return EMBEDDING_MODELS.get(model_name, model_name)


class OllamaEmbeddingClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _endpoint(self) -> str:
        return f"{self.base_url}/api/embed"

    def _request_embedding(self, text: str) -> list[float]:
        response = requests.post(
            url=self._endpoint(),
            json={"model": self.model, "input": text},
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError("Ollama response did not include embeddings.")
        return embeddings[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for text in texts:
            try:
                embeddings.append(self._request_embedding(text=text))
            except Exception:
                logger.exception("Ollama embedding request failed")
                embeddings.append([])

        return embeddings


def get_embedding_client(
    model_name: str | None = None,
    base_url: str | None = None,
) -> OllamaEmbeddingClient:
    resolved_model_name = resolve_embedding_model_name(model_name=model_name)
    resolved_base_url = (base_url or "http://localhost:11434").rstrip("/")
    return OllamaEmbeddingClient(base_url=resolved_base_url, model=resolved_model_name)
