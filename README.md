# Complaint-Analysis-Chatbot-RAG

## Overview

The Complaint Intelligence System is designed to transform large-scale, unstructured consumer complaint data into a clean, structured dataset that can power semantic search and Retrieval-Augmented Generation (RAG) applications.

This repository contains the **data processing and exploratory analysis layer (Task 1)**, focusing on preparing high-quality text data from CFPB consumer complaints.

The goal is to convert raw complaints into a reliable dataset suitable for embedding, retrieval, and downstream AI systems.

---

## Objectives

In this phase, the following steps were completed:

- Loaded and explored a large-scale CFPB complaint dataset
- Performed data quality assessment and missing value analysis
- Filtered relevant financial product categories:
  - Credit Card
  - Personal Loan
  - Savings Account
  - Money Transfer
- Conducted exploratory data analysis (EDA):
  - Product distribution analysis
  - Complaint narrative availability analysis
  - Word count distribution analysis
- Built a text preprocessing pipeline:
  - Lowercasing text
  - Removing special characters and noise
  - Removing URLs and HTML artifacts
  - Removing boilerplate complaint templates
  - Cleaning redacted placeholders (e.g., XXXX)
- Generated a cleaned dataset ready for embedding and vector search

---

## 📁 Project Structure

```text
Complaint-Analysis-Chatbot-RAG/
│
├── .github/
│ └── workflows/
│ └── unittests.yml
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── filtered_complaints.csv
│
├── notebooks/
│ ├── EDA.ipynb
│ └── preprocessing.ipynb
│
├── src/
│ └── preprocessing.py
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

git clone https://github.com/RahemetGisho/Complaint-Analysis-Chatbot-RAG.git

cd Complaint-Analysis-Chatbot-RAG

### 2. Create a virtual environment

python -m venv .venv

Activate:

Windows

.venv\Scripts\activate

Linux / Mac

source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

## Exploratory Data Analysis Summary

### Data Cleaning Pipeline

The preprocessing pipeline includes:

- Text normalization (lowercasing)
- Noise removal (special characters, URLs, HTML artifacts)
- Boilerplate removal (formal complaint templates)
- Redaction cleanup (e.g., masked identifiers like XXXX)
- Whitespace normalization

### Output Dataset

The final processed dataset is saved as:

data/filtered_complaints.csv

This file contains:

- Clean complaint narratives
- Relevant product categories
- Structured metadata for analysis
