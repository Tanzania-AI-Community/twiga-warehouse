from typing import Protocol

from src.config.settings import settings
from src.domain.entities.chunker import EmbedderProvider
from src.infrastructure.embedder.ollama_embedder import get_embedding_client as get_ollama_embedding_client
from src.infrastructure.embedder.together_embedder import get_embedding_client as get_together_embedding_client


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


def resolve_embedding_provider(provider: EmbedderProvider | str | None) -> EmbedderProvider:
    if provider is None:
        return EmbedderProvider.OLLAMA
    if isinstance(provider, EmbedderProvider):
        return provider
    return EmbedderProvider(provider)


def get_embedding_client(
    *,
    provider: EmbedderProvider | str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> EmbeddingClient:
    resolved_provider = resolve_embedding_provider(provider)

    if resolved_provider == EmbedderProvider.OLLAMA:
        return get_ollama_embedding_client(model_name=model_name, base_url=base_url)

    if resolved_provider == EmbedderProvider.TOGETHER:
        return get_together_embedding_client(
            model_name=model_name,
            api_key=settings.TOGETHER_AI_API_KEY,
        )

    raise ValueError(f"Unsupported embedding provider: {resolved_provider}")


def get_embeddings(
    texts: list[str],
    *,
    provider: EmbedderProvider | str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    client = get_embedding_client(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    return client.embed_documents(texts)
