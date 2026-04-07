import hashlib
import logging
import re
from abc import ABC, abstractmethod

import numpy as np

from ._compat import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required to use OpenAIEmbeddingModel. Install project requirements first."
            ) from exc

        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to use SBertEmbeddingModel."
            ) from exc

        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class HashEmbeddingModel(BaseEmbeddingModel):
    """
    Lightweight local embedding model for smoke tests and offline runs.
    """

    def __init__(self, dimension=256):
        if not isinstance(dimension, int) or dimension < 8:
            raise ValueError("dimension must be an integer greater than or equal to 8")
        self.dimension = dimension

    def create_embedding(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())

        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            slot = int(digest, 16) % self.dimension
            sign = 1.0 if int(digest[-1], 16) % 2 == 0 else -1.0
            vector[slot] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector.tolist()
