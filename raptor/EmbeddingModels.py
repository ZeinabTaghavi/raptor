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


class TransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name="facebook/contriever",
        pooling="mean",
        normalize=True,
        max_length=512,
        trust_remote_code=True,
        device=None,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required to use TransformersEmbeddingModel."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(
            1
        ).clamp(min=1e-9)

    def create_embedding(self, text):
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        if self.pooling == "cls":
            embedding = outputs.last_hidden_state[:, 0]
        else:
            embedding = self._mean_pool(outputs, inputs["attention_mask"])
        if self.normalize:
            embedding = self._torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding[0].detach().cpu().numpy().astype(np.float32).tolist()


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
