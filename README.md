# Simple RAG Backend

A lightweight backend API for Retrieval-Augmented Generation (RAG).  
Ingest `.xlsx` and `.docx` files, chunk and embed their content, store it in a vector database, and use a local LLM to
answer questions based on retrieved context.

---

## System Requirements

- **Python**: 3.10+
- **Operating System**: macOS, Windows, or Linux
- **Ollama**: Installed and running with the Llama 3 model pulled
- **RAM**: Minimum 4GB (For larger datasets I recommended 8GB )

---

## Overview

This section outlines the five main stages of the implemented RAG system.
 
This project implements a standard RAG pipeline:

1. **Ingestion** – Upload files via API, parse their content, and extract metadata.
2. **Processing** – Split large texts into manageable chunks.
3. **Embedding & Indexing** – Convert chunks to vector embeddings and store them in FAISS for fast similarity search.
4. **Retrieval** – Search the vector store for the top-k most relevant chunks.
5. **Generation** – Pass retrieved chunks + the user’s query to a local LLM to generate factual, concise answers.

---

## Architecture & Tech Stack

| Component            | Choice & Reason                                                            |
|----------------------|----------------------------------------------------------------------------|
| **Language**         | Python – Fast development, strong ecosystem for ML/NLP.                    |
| **Framework**        | FastAPI – High performance, auto-generated API docs, async-friendly.       |
| **Document Parsing** | `pandas` for Excel, `python-docx` for Word.                                |
| **Embedding Model**  | HuggingFace `all-MiniLM-L6-v2` – Lightweight and local.                    |
| **Vector Store**     | FAISS – High-performance similarity search.                                |
| **LLM Runtime**      | Ollama – Runs Llama 3 locally.                                             |
| **Orchestration**    | LangChain – Abstractions for splitting, embedding, storage, and prompting. |

---

## Installation

### Install & Set Up Ollama

Your server needs a local LLM via [Ollama](https://ollama.com).
Follow the instructions on their website to install Ollama for your OS.
Download & install Ollama (macOS, Windows, or Linux)

After installation, ensure Ollama is running and the Llama 3 model is pulled:
```bash
# Then run Ollama (macOS/Windows auto-start; Linux use:)
ollama serve

# Pull the Llama 3 model
ollama pull llama3
```

### Set Up the Server Environment

- clone the repository at https://github.com/BARDAMRI/RAG_System.git
- Create a virtual environment:

```bash


# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload
```

API base URL: http://localhost:8000  
API docs: http://localhost:8000/docs

---

## Example Usage

### 1. Ingest Documents

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@data/sample.xlsx" \
  -F "files=@data/sample.docx"
```

**Response:**

```json
{
  "message": "Successfully indexed 3 chunks from 2 files"
}
```

### 2. Query the System

**Hebrew Example**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "מי עובד במיקרוסופט?"}'
```

**Output:**

```json
{
  "answer": "על פי המסמכים, צ'רלי הוא מהנדס תוכנה במיקרוסופט.",
  "sources": [
    "sample.xlsx (excel index 2)"
  ]
}
```

**English Example**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 2}'
```

**Output:**

```json
{
  "answer": "RAG, or Retrieval-Augmented Generation, is a system that combines retrieval and language generation to answer queries from stored documents.",
  "sources": [
    "sample.docx (docx index 1)"
  ]
}
```

---

## API Reference

### POST /ingest

- **Description**: Upload one or more `.xlsx` or `.docx` files for ingestion, processing, and indexing.
- **Body**: Multipart form data with one or more `files` fields.
- **Example Request**:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@data/sample.xlsx" \
  -F "files=@data/sample.docx"
```

- **Example Response**:

```json
{
  "message": "Successfully indexed 3 chunks from 2 files"
}
```

- **Possible Errors**:
  - `400 Bad Request`: No files provided or unsupported file format.
  - `500 Internal Server Error`: Error during file processing or indexing.

---

### POST /query

- **Description**: Query the system with a question to receive an answer generated from retrieved document chunks.
- **Body**: JSON object with fields:
  - `question` (string, required): The user's query.
  - `top_k` (integer, optional): Number of top relevant chunks to retrieve (default is 3).
- **Example Request**:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 2}'
```

- **Example Response**:

```json
{
  "answer": "RAG, or Retrieval-Augmented Generation, is a system that combines retrieval and language generation to answer queries from stored documents.",
  "sources": [
    "sample.docx (docx index 1)"
  ]
}
```

- **Possible Errors**:
  - `400 Bad Request`: Missing or invalid question field.
  - `404 Not Found`: No relevant documents found.
  - `500 Internal Server Error`: Error during retrieval or generation.

---

## Testing

Once running:

1. Ingest sample docs via `/ingest`.
2. Query via `/query` and verify the answer matches the document contents.
3. Try varying `top_k` to adjust the number of retrieved chunks.

---

## Troubleshooting / Known Issues

- Ollama not running → Ensure `ollama serve` is active
- FAISS index not found → Ensure documents were ingested successfully
- Model loading slow → Check available RAM and disk speed

---

## Data Persistence

By default, the FAISS index and processed data are stored in the `vectorstore/` directory inside the project root.  
This folder must be preserved to retain ingested data between server restarts.

---

## License

MIT License – Feel free to use and modify.