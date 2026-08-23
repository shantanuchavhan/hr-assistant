import logging
from typing import Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from config import (
    AZURE_OPENAI_GPT_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
)

logger = logging.getLogger(__name__)


class _LazyLLM:
    def __init__(self) -> None:
        self._inst: Any | None = None

    def _init(self) -> None:
        if self._inst is not None:
            return
        try:
            kwargs = {"temperature": 0.2, "max_tokens": 512}
            if AZURE_OPENAI_GPT_DEPLOYMENT:
                kwargs["azure_deployment"] = AZURE_OPENAI_GPT_DEPLOYMENT
            if AZURE_OPENAI_ENDPOINT:
                kwargs["azure_endpoint"] = AZURE_OPENAI_ENDPOINT
            if AZURE_OPENAI_KEY:
                kwargs["api_key"] = AZURE_OPENAI_KEY
            if AZURE_OPENAI_API_VERSION:
                kwargs["api_version"] = AZURE_OPENAI_API_VERSION

            if any(k in kwargs for k in ("azure_deployment", "azure_endpoint", "api_key")):
                self._inst = AzureChatOpenAI(**kwargs)
            else:
                logger.warning("Azure chat config missing — using ChatOpenAI fallback")
                self._inst = ChatOpenAI(temperature=0.2, max_tokens=512)
        except Exception:
            logger.exception("Failed to initialize LLM")
            raise

    def __getattr__(self, name: str):
        self._init()
        assert self._inst is not None
        return getattr(self._inst, name)


llm = _LazyLLM()
