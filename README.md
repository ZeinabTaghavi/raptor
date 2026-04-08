<!-- <p align="center">
  <img align="center" src="raptor.jpg" width="1000px" />
</p>
<p align="left"> -->

<!-- <picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor.jpg" width="1000px">
  <source media="(prefers-color-scheme: light)" srcset="raptor_dark.png" width="1000px">
  
</picture> -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor_dark.png">
  <img alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="raptor.jpg">
</picture>

## RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

**RAPTOR** introduces a novel approach to retrieval-augmented language models by constructing a recursive tree structure from documents. This allows for more efficient and context-aware information retrieval across large texts, addressing common limitations in traditional language models. 



For detailed methodologies and implementations, refer to the original paper:

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2401.18059)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/raptor-recursive-abstractive-processing-for/question-answering-on-quality)](https://paperswithcode.com/sota/question-answering-on-quality?p=raptor-recursive-abstractive-processing-for)

## Installation

Before using RAPTOR, ensure Python 3.8+ is installed. Clone the RAPTOR repository and install necessary dependencies:

```bash
git clone https://github.com/parthsarthi03/raptor.git
cd raptor
pip install -r requirements.txt
```

## Basic Usage

To get started with RAPTOR, follow these steps:

### Setting Up RAPTOR

First, set your OpenAI API key and initialize the RAPTOR configuration:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

from raptor import RetrievalAugmentation

# Initialize with default configuration. For advanced configurations, check the documentation. [WIP]
RA = RetrievalAugmentation()
```

### Adding Documents to the Tree

Add your text documents to RAPTOR for indexing:

```python
with open('sample.txt', 'r') as file:
    text = file.read()
RA.add_documents(text)
```

### Answering Questions

You can now use RAPTOR to answer questions based on the indexed documents:

```python
question = "How did Cinderella reach her happy ending?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)
```

### Saving and Loading the Tree

Save the constructed tree to a specified path:

```python
SAVE_PATH = "demo/cinderella"
RA.save(SAVE_PATH)
```

Load the saved tree back into RAPTOR:

```python
RA = RetrievalAugmentation(tree=SAVE_PATH)
answer = RA.answer_question(question=question)
```

## Standalone Experiment Runner

This repository now includes a standalone dataset runner for producing raw RAPTOR artifacts for later evaluation in another system.

Run it with:

```bash
python scripts/run_raptor_experiment.py \
  --dataset-name <dataset_name> \
  --default-yaml <path/to/default_experiment.yaml> \
  --run-name <run_name>
```

The runner creates:

```text
raptor_runs/
  <dataset_name>/
    <run_name>/
      config/
      selection/
      corpus/
      trees/
      retrieval/
      rag/
      profiling/
      run_manifest.json
```

The saved artifacts include:

- the exact copied input YAML and the fully resolved RAPTOR run config
- selected document and QA entries
- per-document RAPTOR trees plus tree statistics
- leaf chunk exports and node indexes with `descendant_leaf_chunk_ids`
- raw retrieval payloads with expanded chunk mappings
- raw answer-generation outputs
- build/query timings and resource snapshots when available

The runner is designed for per-document retrieval scope on long-document QA datasets. It does not compute gold labels, retrieval metrics, or answer-quality metrics.

### YAML Shape

The runner works best when the reference YAML contains a `raptor_run:` section. It will also try to map a few common defaults such as `split`, `max_docs`, `max_questions`, `top_k`, and chunk/token settings from non-RAPTOR YAML files, and records those assumptions in `run_manifest.json`.

Minimal example:

```yaml
raptor_run:
  dataset:
    documents:
      path: sample.txt
      format: text
      doc_id: my_doc
    qa:
      path: qa.json
      format: json
      records_path: qa_entries
      query_id_field: query_id
      doc_id_field: doc_id
      question_field: question
      reference_answers_field: reference_answers
  models:
    embedding:
      provider: openai
      model: text-embedding-ada-002
    summarization:
      provider: openai
      model: gpt-3.5-turbo
    qa:
      provider: openai
      model: gpt-3.5-turbo
  tree_builder:
    max_tokens: 100
    num_layers: 5
  retrieval:
    top_k: 5
    max_tokens: 3500
    collapse_tree: true
```

A fully local example that does not rely on OpenAI is included at `demo/cinderella_experiment.yaml`.

QASPER is also wired in as a native dataset loader. The RAPTOR-specific config for the 25-document retrieval-ablation slice is at `configs/raptor/qasper_retrieval_ablation.yaml` and can be run with:

```bash
python scripts/run_raptor_experiment.py \
  --dataset-name qasper \
  --default-yaml configs/raptor/qasper_retrieval_ablation.yaml
```

Or via the wrapper script that defaults to GPUs `0,1`:

```bash
bash main.sh \
  --dataset-name qasper \
  --default-yaml configs/raptor/qasper_retrieval_ablation.yaml
```

That QASPER config is set up to avoid OpenAI dependencies by default:

- embeddings use `facebook/contriever`
- tree summarization uses `Qwen/Qwen2-0.5B-Instruct`
- answer generation uses the same `Qwen/Qwen2-0.5B-Instruct` model 

The runner also supports `transformers`-backed local generation models if you prefer not to use `vllm`.

LooGLE is wired in the same way. The RAPTOR-specific configs are:

- `configs/raptor/loogle_retrieval_ablation.yaml`
- `configs/raptor/loogle_retrieval_ablation_transformers.yaml`

Run the default `vllm` variant with:

```bash
python scripts/run_raptor_experiment.py \
  --dataset-name loogle \
  --default-yaml configs/raptor/loogle_retrieval_ablation.yaml
```

Or via the wrapper:

```bash
bash main.sh \
  --dataset-name loogle \
  --default-yaml configs/raptor/loogle_retrieval_ablation.yaml
```

If `vllm` is not workable in your environment, use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_raptor_experiment.py \
  --dataset-name loogle \
  --default-yaml configs/raptor/loogle_retrieval_ablation_transformers.yaml
```

NarrativeQA is supported through the same unified dataset-loader path. The RAPTOR-specific configs are:

- `configs/raptor/nqa_retrieval_ablation.yaml`
- `configs/raptor/nqa_retrieval_ablation_transformers.yaml`

Use `--dataset-name narrativeqa` so its outputs are stored separately under `raptor_runs/narrativeqa/...`.

Run the default `vllm` variant with:

```bash
python scripts/run_raptor_experiment.py \
  --dataset-name narrativeqa \
  --default-yaml configs/raptor/nqa_retrieval_ablation.yaml
```

Or via the wrapper:

```bash
bash main.sh \
  --dataset-name narrativeqa \
  --default-yaml configs/raptor/nqa_retrieval_ablation.yaml
```

If `vllm` is not workable in your environment, use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_raptor_experiment.py \
  --dataset-name narrativeqa \
  --default-yaml configs/raptor/nqa_retrieval_ablation_transformers.yaml
```

QuALITY is supported through the same unified dataset-loader path. The RAPTOR-specific configs are:

- `configs/raptor/quality_retrieval_ablation.yaml`
- `configs/raptor/quality_retrieval_ablation_transformers.yaml`

Use `--dataset-name quality` so its outputs are stored separately under `raptor_runs/quality/...`.
Relative output roots are resolved under this RAPTOR project, even if you pass a reference YAML from another repository.

Run the default `vllm` variant with:

```bash
python scripts/run_raptor_experiment.py \
  --dataset-name quality \
  --default-yaml configs/experiments/quality_retrieval_ablation.yaml
```

The dedicated helper script uses the same QuALITY config:

```bash
bash scripts/run_quality_experiment.sh
```

You can also call the RAPTOR-specific config path directly:

```bash
python scripts/run_raptor_experiment.py \
  --dataset-name quality \
  --default-yaml configs/raptor/quality_retrieval_ablation.yaml
```

Or via the wrapper:

```bash
bash main.sh \
  --dataset-name quality \
  --default-yaml configs/raptor/quality_retrieval_ablation.yaml
```

If `vllm` is not workable in your environment, use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_raptor_experiment.py \
  --dataset-name quality \
  --default-yaml configs/raptor/quality_retrieval_ablation_transformers.yaml
```


### Extending RAPTOR with other Models

RAPTOR is designed to be flexible and allows you to integrate any models for summarization, question-answering (QA), and embedding generation. Here is how to extend RAPTOR with your own models:

#### Custom Summarization Model

If you wish to use a different language model for summarization, you can do so by extending the `BaseSummarizationModel` class. Implement the `summarize` method to integrate your custom summarization logic:

```python
from raptor import BaseSummarizationModel

class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        # Initialize your model here
        pass

    def summarize(self, context, max_tokens=150):
        # Implement your summarization logic here
        # Return the summary as a string
        summary = "Your summary here"
        return summary
```

#### Custom QA Model

For custom QA models, extend the `BaseQAModel` class and implement the `answer_question` method. This method should return the best answer found by your model given a context and a question:

```python
from raptor import BaseQAModel

class CustomQAModel(BaseQAModel):
    def __init__(self):
        # Initialize your model here
        pass

    def answer_question(self, context, question):
        # Implement your QA logic here
        # Return the answer as a string
        answer = "Your answer here"
        return answer
```

#### Custom Embedding Model

To use a different embedding model, extend the `BaseEmbeddingModel` class. Implement the `create_embedding` method, which should return a vector representation of the input text:

```python
from raptor import BaseEmbeddingModel

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        # Initialize your model here
        pass

    def create_embedding(self, text):
        # Implement your embedding logic here
        # Return the embedding as a numpy array or a list of floats
        embedding = [0.0] * embedding_dim  # Replace with actual embedding logic
        return embedding
```

#### Integrating Custom Models with RAPTOR

After implementing your custom models, integrate them with RAPTOR as follows:

```python
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

# Initialize your custom models
custom_summarizer = CustomSummarizationModel()
custom_qa = CustomQAModel()
custom_embedding = CustomEmbeddingModel()

# Create a config with your custom models
custom_config = RetrievalAugmentationConfig(
    summarization_model=custom_summarizer,
    qa_model=custom_qa,
    embedding_model=custom_embedding
)

# Initialize RAPTOR with your custom config
RA = RetrievalAugmentation(config=custom_config)
```

Check out `demo.ipynb` for examples on how to specify your own summarization/QA models, such as Llama/Mistral/Gemma, and Embedding Models such as SBERT, for use with RAPTOR.

Note: More examples and ways to configure RAPTOR are forthcoming. Advanced usage and additional features will be provided in the documentation and repository updates.

## Contributing

RAPTOR is an open-source project, and contributions are welcome. Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

## License

RAPTOR is released under the MIT License. See the LICENSE file in the repository for full details.

## Citation

If RAPTOR assists in your research, please cite it as follows:

```bibtex
@inproceedings{sarthi2024raptor,
    title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
    author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D.},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```

Stay tuned for more examples, configuration guides, and updates.
