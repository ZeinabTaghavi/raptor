import os
from multiprocessing import freeze_support
from pathlib import Path

from raptor import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    TransformersEmbeddingModel,
    VLLMQAModel,
    VLLMSummarizationModel,
)

BASE_DIR = Path(__file__).resolve().parent
QWEN_MODEL = os.environ.get("RAPTOR_QWEN_MODEL", "Qwen/Qwen2-0.5B-Instruct")
CONTRIEVER_MODEL = os.environ.get("RAPTOR_EMBEDDING_MODEL", "facebook/contriever")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

engine_kwargs = {
    "trust_remote_code": True,
    "tensor_parallel_size": int(os.environ.get("RAPTOR_VLLM_TENSOR_PARALLEL_SIZE", "1")),
}
if os.environ.get("RAPTOR_VLLM_DTYPE"):
    engine_kwargs["dtype"] = os.environ["RAPTOR_VLLM_DTYPE"]
if os.environ.get("RAPTOR_VLLM_GPU_MEMORY_UTILIZATION"):
    engine_kwargs["gpu_memory_utilization"] = float(
        os.environ["RAPTOR_VLLM_GPU_MEMORY_UTILIZATION"]
    )

def build_config():
    return RetrievalAugmentationConfig(
        embedding_model=TransformersEmbeddingModel(
            model_name=CONTRIEVER_MODEL,
            trust_remote_code=True,
            device=os.environ.get("RAPTOR_EMBEDDING_DEVICE"),
        ),
        summarization_model=VLLMSummarizationModel(
            model_name=QWEN_MODEL,
            engine_kwargs=engine_kwargs,
            temperature=0.0,
            top_p=1.0,
        ),
        qa_model=VLLMQAModel(
            model_name=QWEN_MODEL,
            engine_kwargs=engine_kwargs,
            default_max_tokens=64,
            temperature=0.0,
            top_p=1.0,
        ),
        tb_max_tokens=80,
        tb_num_layers=2,
        tb_summarization_length=60,
        tr_top_k=4,
    )


def main():
    with open(BASE_DIR / "demo" / "sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Embedding model: {CONTRIEVER_MODEL}")
    print(f"Generation model: {QWEN_MODEL}")
    print(f"Generation backend: vllm ({engine_kwargs})")

    ra = RetrievalAugmentation(config=build_config())
    ra.add_documents(text)

    payload = ra.answer_question_with_metadata(
        "How did Cinderella reach her happy ending?"
    )
    print(payload.keys())
    print(payload["retrieval"].keys())
    print(payload["retrieval"]["retrieved_nodes"][0].keys())
    print(payload["answer"])


if __name__ == "__main__":
    freeze_support()
    main()
