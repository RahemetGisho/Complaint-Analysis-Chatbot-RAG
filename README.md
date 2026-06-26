# Complaint-Analysis-Chatbot-RAG

A Retrieval-Augmented Generation (RAG) tool that lets Product, Support, and Compliance staff ask plain-English questions about customer complaints and get answers grounded in real complaint excerpts — every claim traceable back to its source.

Built on the CFPB Consumer Complaint Database, scoped to four product lines: Credit Card, Personal Loan, Savings Account, and Money Transfer.

## How it works

1. **Data preparation** — CFPB complaints are filtered to the target products and cleaned (PII-redaction tokens and boilerplate removed), then split into chunks.
2. **Embedding & indexing** — Each chunk is embedded with `all-MiniLM-L6-v2` and stored in a ChromaDB vector store, with metadata (complaint ID, product, company, etc.) attached for traceability.
3. **Retrieval-augmented generation** — A question is embedded with the same model, the most relevant chunks are retrieved, and an LLM generates an answer using only that retrieved context.
4. **Interactive UI** — A Gradio interface for non-technical staff: ask a question, watch the answer stream in, and see the exact source excerpts behind it.

## Project structure

```
Complaint-Analysis-Chatbot-RAG/
├── app.py
├── src/
│   ├── config.py
│   ├── sampling.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── vector_store_loader.py
│   ├── retriever.py
│   ├── prompt.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── evaluate.py
├── notebooks/
├── data/
│   ├── raw/
│   └── processed
├── vector_store/
├── reports/
├── tests/
├── requirements.txt
└── .env
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

## Data

Two inputs are required and are not committed to the repository:

| File                           | Description                           | Used by                             |
| ------------------------------ | ------------------------------------- | ----------------------------------- |
| Full CFPB complaint export     | Raw complaint data                    | `notebooks/eda_preprocessing.ipynb` |
| `complaint_embeddings.parquet` | Pre-built chunk embeddings + metadata | `vector_store_loader.py`            |

Place the parquet file at `data/raw/complaint_embeddings.parquet` before first run.

## Usage

Build/refresh the vector store and run the evaluation harness:

```bash
python src/evaluate.py
```

Launch the chat interface:

```bash
python app.py
```

Then open the local URL Gradio prints (default `http://127.0.0.1:7860`).

## Configuration

Key settings in `src/config.py`:

| Setting                  | Default                    | Purpose                                |
| ------------------------ | -------------------------- | -------------------------------------- |
| `TOP_K`                  | `5`                        | Chunks retrieved per question          |
| `GENERATION_BACKEND`     | `hf_inference_api`         | `hf_inference_api` or `local_pipeline` |
| `HF_GENERATION_MODEL`    | `Qwen/Qwen2.5-7B-Instruct` | Hosted model (requires `HF_TOKEN`)     |
| `LOCAL_GENERATION_MODEL` | `google/flan-t5-base`      | Local fallback, no API key needed      |
| `COLLECTION_NAME`        | `complaints`               | ChromaDB collection name               |

## Testing

```bash
pytest tests/ -v
```

## Tech stack

Python · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · Hugging Face Inference API / Transformers · Gradio · LangChain text splitters
