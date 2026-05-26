import logging

from together import Together


logger = logging.getLogger(__name__)


EMBEDDING_MODELS = {
    "multilingual-large": "intfloat/multilingual-e5-large-instruct",
}
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS["multilingual-large"]


def resolve_embedding_model_name(model_name: str | None) -> str:
    if model_name is None:
        return DEFAULT_EMBEDDING_MODEL
    return EMBEDDING_MODELS.get(model_name, model_name)


class TogetherEmbeddingClient:
    def __init__(self, api_key: str, model: str):
        self.client = Together(api_key=api_key)
        self.model = model

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        if response.data is None:
            raise ValueError("Together response did not include embeddings.")

        ordered_data = sorted(response.data, key=lambda item: item.index)
        embeddings: list[list[float]] = []

        for item in ordered_data:
            if item.embedding is None:
                raise ValueError("Together response included an empty embedding.")
            embeddings.append(item.embedding)

        if len(embeddings) != len(texts):
            raise ValueError("Together response embedding count did not match the input size.")

        return embeddings

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        embeddings: list[list[float]] = []

        for start_index in range(0, len(texts), batch_size):
            batch = texts[start_index : start_index + batch_size]

            try:
                embeddings.extend(self._request_embeddings(texts=batch))
            except Exception:
                logger.exception("Together embedding request failed")
                embeddings.extend([[] for _ in batch])

        return embeddings


def get_embedding_client(
    model_name: str | None = None,
    api_key: str | None = None,
) -> TogetherEmbeddingClient:
    if not api_key:
        raise ValueError("TOGETHER_AI_API_KEY is required when embedding_provider='together'.")

    resolved_model_name = resolve_embedding_model_name(model_name=model_name)
    return TogetherEmbeddingClient(api_key=api_key, model=resolved_model_name)
