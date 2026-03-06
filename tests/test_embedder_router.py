from src.domain.entities.chunker import EmbedderProvider
from src.infrastructure.embedder.embedding_router import get_embedding_client, get_embeddings


class _FakeEmbedderClient:
    def __init__(self, value: list[list[float]]):
        self._value = value

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._value


def test_get_embedding_client_selects_together(monkeypatch) -> None:
    captured: dict[str, str | None] = {}
    sentinel = object()

    def _fake_together_client(*, model_name: str | None = None, api_key: str | None = None):
        captured["model_name"] = model_name
        captured["api_key"] = api_key
        return sentinel

    monkeypatch.setattr(
        "src.infrastructure.embedder.embedding_router.get_together_embedding_client",
        _fake_together_client,
    )

    client = get_embedding_client(
        provider=EmbedderProvider.TOGETHER,
        model_name="intfloat/multilingual-e5-large-instruct",
    )

    assert client is sentinel
    assert captured["model_name"] == "intfloat/multilingual-e5-large-instruct"


def test_get_embeddings_defaults_to_ollama(monkeypatch) -> None:
    expected = [[0.1, 0.2, 0.3]]

    def _fake_get_embedding_client(*args, **kwargs):
        return _FakeEmbedderClient(expected)

    monkeypatch.setattr(
        "src.infrastructure.embedder.embedding_router.get_embedding_client",
        _fake_get_embedding_client,
    )

    result = get_embeddings(["hello"])
    assert result == expected
