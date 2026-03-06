from src.infrastructure.embedder.together_embedder import (
    DEFAULT_EMBEDDING_MODEL,
    get_embedding_client,
)


class _FakeTogetherEmbeddings:
    def __init__(self, *, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def embed_query(self, text: str):
        return [0.0]

    def embed_documents(self, texts: list[str]):
        return [[float(i)] for i, _ in enumerate(texts)]


def test_get_embedding_client_uses_together_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.embedder.together_embedder.TogetherEmbeddings",
        _FakeTogetherEmbeddings,
    )

    client = get_embedding_client(api_key="test-key")

    assert isinstance(client.client, _FakeTogetherEmbeddings)
    assert client.client.model == DEFAULT_EMBEDDING_MODEL
    assert client.client.api_key == "test-key"


def test_get_embedding_client_resolves_model_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.embedder.together_embedder.TogetherEmbeddings",
        _FakeTogetherEmbeddings,
    )

    client = get_embedding_client(model_name="multilingual-large", api_key="k")

    assert client.client.model == "intfloat/multilingual-e5-large-instruct"
