import logging
import os
import re

from abc import ABC, abstractmethod

from ._compat import retry, stop_after_attempt, wait_random_exponential
from ._generation_backends import (generate_with_transformers,
                                   generate_with_vllm)


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required to use GPT3QAModel. Install project requirements first."
            ) from exc

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required to use GPT3TurboQAModel. Install project requirements first."
            ) from exc

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            stop=stop_sequence,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required to use GPT4QAModel. Install project requirements first."
            ) from exc

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            stop=stop_sequence,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required to use UnifiedQAModel."
            ) from exc

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]


class ExtractiveQAModel(BaseQAModel):
    """
    Local heuristic QA model that returns the most question-overlapping sentence.
    """

    def __init__(self, model_name="extractive-overlap"):
        self.model = model_name

    def answer_question(self, context, question):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", context)
            if sentence.strip()
        ]

        if not sentences:
            return ""

        question_terms = set(re.findall(r"\w+", question.lower()))
        if not question_terms:
            return sentences[0]

        def score(sentence):
            sentence_terms = set(re.findall(r"\w+", sentence.lower()))
            return len(question_terms & sentence_terms), -len(sentence)

        best_sentence = max(sentences, key=score)
        return best_sentence


class TransformersQAModel(BaseQAModel):
    def __init__(
        self,
        model_name,
        *,
        pipeline_kwargs=None,
        default_max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        stop=None,
    ):
        self.model = model_name
        self.pipeline_kwargs = dict(pipeline_kwargs or {})
        self.default_max_tokens = default_max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def _build_prompt(self, context, question):
        return (
            "You are a question answering assistant. Answer the question using only the given context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def answer_question(
        self,
        context,
        question,
        max_tokens=None,
        stop_sequence=None,
        temperature=None,
        top_p=None,
    ):
        return generate_with_transformers(
            model_name=self.model,
            prompt=self._build_prompt(context, question),
            pipeline_kwargs=self.pipeline_kwargs,
            max_tokens=int(max_tokens or self.default_max_tokens),
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            stop=stop_sequence or self.stop,
        )


class VLLMQAModel(BaseQAModel):
    def __init__(
        self,
        model_name,
        *,
        engine_kwargs=None,
        default_max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        stop=None,
    ):
        self.model = model_name
        self.engine_kwargs = dict(engine_kwargs or {})
        self.default_max_tokens = default_max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def _build_prompt(self, context, question):
        return (
            "You are a question answering assistant. Answer the question using only the given context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def answer_question(
        self,
        context,
        question,
        max_tokens=None,
        stop_sequence=None,
        temperature=None,
        top_p=None,
    ):
        return generate_with_vllm(
            model_name=self.model,
            prompt=self._build_prompt(context, question),
            engine_kwargs=self.engine_kwargs,
            max_tokens=int(max_tokens or self.default_max_tokens),
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            stop=stop_sequence or self.stop,
        )
