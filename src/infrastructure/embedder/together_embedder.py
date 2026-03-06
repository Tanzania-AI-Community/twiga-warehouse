import logging

from langchain_together.embeddings import TogetherEmbeddings


logger = logging.getLogger(__name__)


EMBEDDING_MODELS = {
    "multilingual-large": "intfloat/multilingual-e5-large-instruct",  # 1024 dimensions
}
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS["multilingual-large"]


def resolve_embedding_model_name(model_name: str | None) -> str:
    if model_name is None:
        return DEFAULT_EMBEDDING_MODEL
    return EMBEDDING_MODELS.get(model_name, model_name)


class TogetherEmbedder:
    def __init__(self, api_key: str, model: str):
        self.client = TogetherEmbeddings(
            model=model,
            api_key=api_key,
        )

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.client.embed_documents(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                "Together response embedding count did not match input size "
                f"({len(embeddings)} != {len(texts)})."
            )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(text)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        from tqdm import tqdm

        embeddings: list[list[float]] = []

        for index in tqdm(range(0, len(texts), batch_size)):
            batch = texts[index:index + batch_size]
            try:
                embeddings.extend(self._request_embeddings(batch))
            except Exception:
                logger.error("Together embedding request failed", exc_info=True)
                embeddings.extend([[] for _ in batch])

        return embeddings


def get_embedding_client(
    model_name: str | None = None,
    api_key: str | None = None,
) -> TogetherEmbedder:
    if not api_key:
        raise ValueError("TOGETHER_AI_API_KEY is required when embedding_provider='together'")

    resolved_model = resolve_embedding_model_name(model_name)
    return TogetherEmbedder(api_key=api_key, model=resolved_model)
