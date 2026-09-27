from unittest.mock import MagicMock

from src.models.processing import EmbedderProvider
from src.providers.embedder import router


def test_get_embedding_client_selects_together(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(router, "get_together_embedding_client", factory)
    result = router.get_embedding_client(EmbedderProvider.TOGETHER, "legacy-model")
    assert result is factory.return_value
    assert factory.call_args.kwargs["model_name"] == "legacy-model"


def test_get_embedding_client_selects_google_without_api_key(monkeypatch):
    from src.providers.embedder import google_embedder

    factory = MagicMock()
    monkeypatch.setattr(google_embedder, "GoogleEmbeddingClient", factory)
    monkeypatch.setattr(router.settings, "GOOGLE_CLOUD_PROJECT", "test-project")
    result = router.get_embedding_client(EmbedderProvider.GOOGLE)
    assert result is factory.return_value
    factory.assert_called_once_with(
        project="test-project",
        location="global",
        model="gemini-embedding-001",
        dimensions=1024,
    )
