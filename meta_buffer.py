from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc

if TYPE_CHECKING:
    import numpy as np


class MetaBuffer:
    """A buffer for managing meta-learning and retrieval operations."""

    def __init__(
        self,
        llm_model: str,
        embedding_model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/",
        rag_dir: str = "./test",
    ) -> None:
        """
        Initialize MetaBuffer with model and configuration parameters.

        Args:
            llm_model: Language model identifier
            embedding_model: Embedding model identifier
            api_key: Optional API key for authentication
            base_url: Base URL for API endpoints
            rag_dir: Directory for RAG storage

        """
        self.api_key = api_key
        self.llm = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        rag_path = Path(rag_dir)
        if not rag_path.exists():
            rag_path.mkdir(parents=True, exist_ok=True)
        self.rag = LightRAG(
            working_dir=rag_dir,
            llm_model_func=self.llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=self.embedding_func,
            ),
        )

    async def llm_model_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: dict[str, str | int | float | bool],
    ) -> str:
        if history_messages is None:
            history_messages = []
        return await openai_complete_if_cache(
            self.llm,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        return await openai_embedding(
            texts,
            model=self.embedding_model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def retrieve_and_instantiate(self, query_text: str) -> str:
        return self.rag.query(query_text, param=QueryParam(mode="hybrid"))

    def dynamic_update(self, thought_template: str) -> None:
        from prompts import SIMILARITY_CHECK_PROMPT
        query_text = SIMILARITY_CHECK_PROMPT.format(thought_template=thought_template)
        response = self.rag.query(query_text, param=QueryParam(mode="hybrid"))
        if self.extract_similarity_decision(response):
            self.rag.insert(thought_template)
        else:
            pass

    def extract_similarity_decision(self, text: str) -> bool:
        """
        Extract similarity decision from response text.

        Parse the input text and determine if it indicates similarity
        between templates based on True/False markers.

        Args:
            text: Response text to analyze

        Returns:
            bool: True if templates are similar, False otherwise

        Raises:
            ValueError: If no valid True/False conclusion is found

        """
        # Convert the text to lowercase for easier matching
        text = text.lower()

        # Look for the conclusion part where the decision is made
        if "false" in text:
            return False
        if "true" in text:
            return True
        # In case no valid conclusion is found
        msg = "No valid conclusion (True/False) found in the text."
        raise ValueError(msg)
