from typing import Protocol

from src.config.settings import settings
from src.models.processing import EmbedderProvider
from src.providers.embedder.ollama_embedder import get_embedding_client as get_ollama_embedding_client
from src.providers.embedder.together_embedder import get_embedding_client as get_together_embedding_client


class EmbedderClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


def get_embedding_client(
    provider: EmbedderProvider,
    model_name: str | None = None,
    base_url: str | None = None,
) -> EmbedderClient:
    if provider == EmbedderProvider.OLLAMA:
        return get_ollama_embedding_client(
            model_name=model_name,
            base_url=base_url,
        )

    if provider == EmbedderProvider.TOGETHER:
        return get_together_embedding_client(
            model_name=model_name,
            api_key=settings.TOGETHER_AI_API_KEY,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
