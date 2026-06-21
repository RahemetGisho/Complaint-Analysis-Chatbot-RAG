# Customer Complaint Analysis with RAG

## Overview

This project builds the foundation of a Retrieval-Augmented Generation (RAG) system using customer complaint data from financial products.

- **1:** Exploratory Data Analysis (EDA) and preprocessing
- **2:** Sampling, chunking, embedding generation, and vector store indexing

The final output is a retrieval-ready vector database that can be used for semantic search and downstream RAG applications.

---

# 1: EDA and Preprocessing

## Objectives

- Explore complaint distributions
- Analyze missing values
- Clean complaint narratives
- Standardize product categories
- Prepare data for retrieval and embedding

## Processing Steps

1. Loaded the CFPB complaint dataset
2. Removed records with missing complaint narratives
3. Cleaned complaint text:
   - Lowercasing
   - Removing special characters
   - Removing extra whitespace
4. Standardized product categories
5. Created a filtered dataset for downstream processing

## Output

```text
data/filtered_complaints.csv
```

---

# 2: Text Chunking, Embedding, and Vector Store Indexing

## Objectives

Build a retrieval-ready knowledge base from complaint narratives.

### Pipeline

1. Load cleaned complaint data
2. Draw a proportional stratified sample
3. Chunk complaint narratives
4. Generate embeddings
5. Store embeddings in ChromaDB

---

## Sampling Strategy

### Stratified Sampling

The full dataset contains more than one million complaints. Generating embeddings for the entire dataset can be computationally expensive.

Using a stratified sample:

- Preserves category balance
- Reduces processing time
- Demonstrates the complete RAG pipeline efficiently

---

## Chunking Strategy

Complaint narratives are split using LangChain's `RecursiveCharacterTextSplitter`.

### Parameters

```python
chunk_size = 500
chunk_overlap = 50
```

---

## Embedding Model

Model used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## Vector Store

Embeddings are stored in ChromaDB.

---

# Project Structure

```text
Complaint-Analysis-Chatbot-RAG/
│
├── README.md
│
├── data/
│   ├── raw
│       ├── filtered_complaints.csv
│   │
│   └── processed/
│       ├── sampled_complaints.csv
│       └── chunks_metadata.csv
│
├── notebooks/
│   └── eda_preprocessing.ipynb
│
├── src/
│   ├── sampling.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── build_pipeline.py
│
├── vector_store/
│
├── reports/
│
├── requirements.txt
│
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/RahemetGisho/Complaint-Analysis-Chatbot-RAG.git
cd Complaint-Analysis-Chatbot-RAG
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running Task 2 Pipeline

From the project root:

```bash
python src/build_pipeline.py
```

---

# Generated Outputs

## Stratified Sample

```text
data/processed/sampled_complaints.csv
```

Contains the 12,000-row proportional stratified sample.

---

## Chunk Metadata

```text
data/processed/chunks_metadata.csv
```

Contains all generated chunks and associated metadata for inspection and debugging.

---

## ChromaDB Vector Store

```text
vector_store/
```

Contains persisted embeddings and metadata used for semantic retrieval.

---

# Technologies Used

- Python
- Pandas
- NumPy
- LangChain
- Sentence Transformers
- ChromaDB
- Jupyter Notebook

---
