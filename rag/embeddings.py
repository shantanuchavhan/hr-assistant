import logging
from typing import Any

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from config import (
    AZURE_OPENAI_EMBEDDING_MODEL,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
)

logger = logging.getLogger(__name__)


class _LazyEmbeddings:
    """Lazily instantiate AzureOpenAIEmbeddings to avoid import-time failures.

    This delays client construction until the first attribute access and logs
    a helpful error if initialization fails.
    """

    def __init__(self) -> None:
        self._inst: Any | None = None

    def _init(self) -> None:
        if self._inst is not None:
            return
        try:
            kwargs = {}
            if AZURE_OPENAI_EMBEDDING_MODEL:
                kwargs["azure_deployment"] = AZURE_OPENAI_EMBEDDING_MODEL
            if AZURE_OPENAI_ENDPOINT:
                kwargs["azure_endpoint"] = AZURE_OPENAI_ENDPOINT
            if AZURE_OPENAI_KEY:
                kwargs["api_key"] = AZURE_OPENAI_KEY
            if AZURE_OPENAI_API_VERSION:
                kwargs["api_version"] = AZURE_OPENAI_API_VERSION

            # Prefer Azure embeddings when configuration is present
            logger.debug("Initializing AzureOpenAIEmbeddings with kwargs: %r", kwargs)
            if kwargs:
                try:
                    self._inst = AzureOpenAIEmbeddings(**kwargs)
                except Exception:
                    logger.exception("AzureOpenAIEmbeddings init failed with kwargs: %r", kwargs)
                    # Fallback to generic OpenAIEmbeddings to allow the app to continue.
                    try:
                        logger.warning("Falling back to OpenAIEmbeddings due to Azure init failure")
                        self._inst = OpenAIEmbeddings()
                    except Exception:
                        logger.exception("Fallback OpenAIEmbeddings init also failed")
                        raise
            else:
                # Fall back to a generic OpenAIEmbeddings instance (no creds)
                logger.warning("Azure config missing — using default OpenAIEmbeddings fallback")
                self._inst = OpenAIEmbeddings()
        except Exception:
            logger.exception("Failed to initialize embeddings")
            # Re-raise to make the failure explicit to callers that need embeddings
            raise

    def __getattr__(self, name: str):
        self._init()
        assert self._inst is not None
        return getattr(self._inst, name)


embeddings = _LazyEmbeddings()
