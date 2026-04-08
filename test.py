import os
from pathlib import Path

from raptor import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    TransformersEmbeddingModel,
    TransformersQAModel,
    TransformersSummarizationModel,
)

BASE_DIR = Path(__file__).resolve().parent
QWEN_MODEL = os.environ.get("RAPTOR_QWEN_MODEL", "Qwen/Qwen2-0.5B-Instruct")
CONTRIEVER_MODEL = os.environ.get("RAPTOR_EMBEDDING_MODEL", "facebook/contriever")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# Keep the smoke test CPU-friendly by default. Users with accelerate/GPU support
# can override this with RAPTOR_GENERATION_DEVICE_MAP=auto.
generation_device_map = os.environ.get("RAPTOR_GENERATION_DEVICE_MAP")
pipeline_kwargs = {"device_map": generation_device_map} if generation_device_map else {"device_map": None}

with open(BASE_DIR / "demo" / "sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

config = RetrievalAugmentationConfig(
    embedding_model=TransformersEmbeddingModel(
        model_name=CONTRIEVER_MODEL,
        trust_remote_code=True,
        device=os.environ.get("RAPTOR_EMBEDDING_DEVICE"),
    ),
    summarization_model=TransformersSummarizationModel(
        model_name=QWEN_MODEL,
        pipeline_kwargs=pipeline_kwargs,
        temperature=0.0,
        top_p=1.0,
    ),
    qa_model=TransformersQAModel(
        model_name=QWEN_MODEL,
        pipeline_kwargs=pipeline_kwargs,
        default_max_tokens=64,
        temperature=0.0,
        top_p=1.0,
    ),
    tb_max_tokens=80,
    tb_num_layers=2,
    tb_summarization_length=60,
    tr_top_k=4,
)

print(f"Embedding model: {CONTRIEVER_MODEL}")
print(f"Generation model: {QWEN_MODEL}")

ra = RetrievalAugmentation(config=config)
ra.add_documents(text)

# question = "How did Cinderella reach her happy ending?"
# context, layer_info = ra.retrieve(question, return_layer_information=True)
# answer = ra.answer_question(question)


# ra.save("demo/test_tree.pkl")
# ra2 = RetrievalAugmentation(tree="demo/test_tree.pkl")
# print(ra2.answer_question("How did Cinderella reach her happy ending?"))


# print("Context preview:", context[:500])
# print("Layer info sample:", layer_info[:5])
# print("Answer:", answer)

payload = ra.answer_question_with_metadata("How did Cinderella reach her happy ending?")
print(payload.keys())
print(payload["retrieval"].keys())
print(payload["retrieval"]["retrieved_nodes"][0].keys())
print(payload["answer"])
