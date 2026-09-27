from unittest.mock import MagicMock

import pytest

from src.models import ChunkerType, EmbedderProvider, TextChunk
from src.models.book import ProcessingOptions
from src.pipeline import runner


def test_default_embedding_settings_are_google():
    options = ProcessingOptions(chunker_type=ChunkerType.LANGCHAIN)
    assert options.embedding_provider == EmbedderProvider.GOOGLE
    assert runner.DEFAULT_EMBEDDING_PROVIDER == EmbedderProvider.GOOGLE
    assert runner.DEFAULT_EMBEDDING_MODEL == "gemini-embedding-001"


@pytest.mark.parametrize("vectors", [[], [[1.0]], [[1.0], []]])
def test_pipeline_rejects_partial_embedding_results(monkeypatch, vectors):
    client = MagicMock()
    client.embed_documents.return_value = vectors
    monkeypatch.setattr(runner, "get_embedding_client", lambda **kwargs: client)
    chunks = [
        TextChunk(content=text, page_number=1, chapter_number=1) for text in ["a", "b"]
    ]
    with pytest.raises(ValueError):
        runner.create_embedded_chunks(
            EmbedderProvider.GOOGLE, "gemini-embedding-001", chunks
        )
    client.close.assert_called_once()


def test_pipeline_preserves_chunk_metadata(monkeypatch):
    client = MagicMock()
    client.embed_documents.return_value = [[1.0] * 1024, [2.0] * 1024]
    monkeypatch.setattr(runner, "get_embedding_client", lambda **kwargs: client)
    chunks = [
        TextChunk(content=text, page_number=i, chapter_number=2)
        for i, text in enumerate(["a", "b"], 1)
    ]
    result = runner.create_embedded_chunks(
        EmbedderProvider.GOOGLE, "gemini-embedding-001", chunks
    )
    assert [c.content for c in result] == ["a", "b"]
    assert [c.page_number for c in result] == [1, 2]
    assert all(c.chapter_number == 2 and len(c.embedding) == 1024 for c in result)


def test_output_declares_document_embedding_policy():
    from src.models import TableOfContents

    request = MagicMock()
    request.processing = ProcessingOptions(
        chunker_type=ChunkerType.LANGCHAIN,
        embedding_model_name="gemini-embedding-001",
    )
    payload = runner.build_output_payload(request, TableOfContents(chapters=[]), [])
    assert payload["embedding_metadata"] == {
        "provider": "google",
        "model": "gemini-embedding-001",
        "dimensions": 1024,
        "task_type": "RETRIEVAL_DOCUMENT",
        "document_max_bytes": 2048,
    }
