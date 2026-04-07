import logging
from abc import ABC, abstractmethod

from ._compat import retry, stop_after_attempt, wait_random_exponential
from ._generation_backends import (generate_with_transformers,
                                   generate_with_vllm, warm_vllm_engine)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class ExtractiveSummarizationModel(BaseSummarizationModel):
    """
    Deterministic local summarizer that truncates to the first `max_tokens` tokens.
    """

    def summarize(self, context, max_tokens=150):
        tokens = context.split()
        if max_tokens is None or max_tokens <= 0:
            return context
        return " ".join(tokens[:max_tokens])


class TransformersSummarizationModel(BaseSummarizationModel):
    def __init__(
        self,
        model_name,
        *,
        pipeline_kwargs=None,
        temperature=0.0,
        top_p=1.0,
        stop=None,
    ):
        self.model = model_name
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def summarize(self, context, max_tokens=150, stop_sequence=None):
        prompt = (
            "Write a concise summary of the following text while preserving the key details.\n\n"
            f"Text:\n{context}\n\n"
            "Summary:"
        )
        return generate_with_transformers(
            model_name=self.model,
            prompt=prompt,
            pipeline_kwargs=self.pipeline_kwargs,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop_sequence or self.stop,
        )


class VLLMSummarizationModel(BaseSummarizationModel):
    def __init__(
        self,
        model_name,
        *,
        engine_kwargs=None,
        temperature=0.0,
        top_p=1.0,
        stop=None,
    ):
        self.model = model_name
        self.engine_kwargs = dict(engine_kwargs or {})
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def warm_up(self):
        warm_vllm_engine(model_name=self.model, engine_kwargs=self.engine_kwargs)

    def summarize(self, context, max_tokens=150, stop_sequence=None):
        prompt = (
            "Write a concise summary of the following text while preserving the key details.\n\n"
            f"Text:\n{context}\n\n"
            "Summary:"
        )
        return generate_with_vllm(
            model_name=self.model,
            prompt=prompt,
            engine_kwargs=self.engine_kwargs,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop_sequence or self.stop,
        )
