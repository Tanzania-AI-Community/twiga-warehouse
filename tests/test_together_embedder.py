from src.infrastructure.embedder.together_embedder import (
    DEFAULT_EMBEDDING_MODEL,
    TogetherEmbedder,
    get_embedding_client,
)


class _FakeEmbeddingData:
    def __init__(self, index: int, embedding: list[float]):
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: list[_FakeEmbeddingData]):
        self.data = data


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, *, model: str, input: list[str]) -> _FakeEmbeddingResponse:
        self.calls.append({"model": model, "input": input})
        data = [_FakeEmbeddingData(index=i, embedding=[float(i)]) for i, _ in enumerate(input)]
        return _FakeEmbeddingResponse(data=data)


class _FakeTogetherClient:
    instances: list["_FakeTogetherClient"] = []

    def __init__(self, *, api_key: str):
        self.api_key = api_key
        self.embeddings = _FakeEmbeddingsAPI()
        self.__class__.instances.append(self)


def test_get_embedding_client_uses_default_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.embedder.together_embedder.Together",
        _FakeTogetherClient,
    )

    client = get_embedding_client(api_key="test-key")

    assert isinstance(client, TogetherEmbedder)
    fake_client = _FakeTogetherClient.instances[-1]
    assert fake_client.api_key == "test-key"
    assert client.model == DEFAULT_EMBEDDING_MODEL


def test_get_embedding_client_resolves_model_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.embedder.together_embedder.Together",
        _FakeTogetherClient,
    )

    client = get_embedding_client(model_name="multilingual-large", api_key="k")

    assert client.model == "intfloat/multilingual-e5-large-instruct"


def test_embed_documents_calls_together_in_batches(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.embedder.together_embedder.Together",
        _FakeTogetherClient,
    )

    client = get_embedding_client(api_key="test-key")
    fake_client = _FakeTogetherClient.instances[-1]

    embeddings = client.embed_documents(["a", "b", "c"], batch_size=2)

    assert embeddings == [[0.0], [1.0], [0.0]]
    assert len(fake_client.embeddings.calls) == 2
    assert fake_client.embeddings.calls[0]["input"] == ["a", "b"]
    assert fake_client.embeddings.calls[1]["input"] == ["c"]
