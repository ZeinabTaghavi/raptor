import re


class SimpleTokenizer:
    """Fallback tokenizer when `tiktoken` is unavailable."""

    def __init__(self, name="simple_whitespace") -> None:
        self.name = name

    def encode(self, text: str):
        if not text:
            return []
        return re.findall(r"\S+", text)

    def __repr__(self) -> str:
        return f"SimpleTokenizer(name={self.name!r})"


def get_default_tokenizer():
    try:
        import tiktoken
    except ImportError:
        return SimpleTokenizer()

    return tiktoken.get_encoding("cl100k_base")


try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ImportError:
    def retry(*args, **kwargs):
        def decorator(function):
            return function

        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_random_exponential(*args, **kwargs):
        return None
