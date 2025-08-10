from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import io
import os
import pandas as pd
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

app = FastAPI(
    title="Simple RAG Backend",
    description="Demonstration implementation of a retrieval-augmented generation system."
)

# Initialize global variables
documents = []
vector_store = None
INDEX_PATH = "faiss_index"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="llama3")  # Requires Ollama running locally with llama3 model

# Load FAISS index if it exists
if os.path.exists(INDEX_PATH):
    try:
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")


def process_excel(file_content: io.BytesIO, source_name: str) -> tuple[List[str], List[dict]]:
    """
    Process an Excel file and extract text content and metadata.
    """
    try:
        df = pd.read_excel(file_content)
        texts = []
        metas = []
        for idx, row in df.iterrows():
            text = " ".join(str(val) for val in row.values if pd.notna(val))
            texts.append(text)
            metas.append({"source": source_name, "index": idx + 2, "type": "excel"})
        return texts, metas
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail=f"Invalid Excel file format: {source_name}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing Excel file {source_name}: {exc}")


def process_docx(file_content: io.BytesIO, source_name: str) -> tuple[List[str], List[dict]]:
    """
    Process a Word document and extract text content and metadata.
    """
    try:
        doc = DocxDocument(file_content)
        texts = []
        metas = []
        full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        combined_text = "\n".join(full_text)
        chunks = text_splitter.split_text(combined_text)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metas.append({"source": source_name, "index": i + 1, "type": "docx"})
        return texts, metas
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error processing Word file {source_name}: {exc}")


@app.post("/ingest", summary="Ingest files into the vector store")
async def ingest_files(files: List[UploadFile] = File(...)):
    """
    Accepts a list of uploaded files (Excel or Word) and ingests their contents into the vector store.
    """
    global vector_store, documents
    total_chunks = 0
    for uploaded in files:
        filename = uploaded.filename or "unknown"
        try:
            content = await uploaded.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read file {filename}: {exc}")

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".xlsx":
            texts, metas = process_excel(io.BytesIO(content), source_name=filename)
        elif ext == ".docx":
            texts, metas = process_docx(io.BytesIO(content), source_name=filename)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {filename}")

        for text, meta in zip(texts, metas):
            documents.append({"text": text, "metadata": meta})
        total_chunks += len(texts)

    if documents:
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        if vector_store is None:
            vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        else:
            vector_store.add_texts(texts, metadatas=metadatas)
        # Save the FAISS index to disk
        vector_store.save_local(INDEX_PATH)
        # Clear documents to save memory
        documents.clear()

    return {"message": f"Successfully indexed {total_chunks} chunks from {len(files)} files"}


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.post("/query", response_model=QueryResponse, summary="Query the vector store and generate an answer")
async def query_rag(request: QueryRequest):
    """
    Given a user question, retrieve the most relevant text fragments and generate a response using an LLM.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Vector store is empty. Please ingest documents first.")

    # Retrieve relevant documents
    docs = vector_store.similarity_search(request.question, k=request.top_k or 3)

    # Prepare context for LLM
    context = "\n".join(doc.page_content for doc in docs)
    sources = [
        f"{doc.metadata.get('source', 'unknown source')} ({doc.metadata.get('type', 'unknown')} index {doc.metadata.get('index', 'N/A')})"
        for doc in docs]

    # Define prompt for LLM
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"You are a helpful assistant. Answer the user's question only based on the following context. If "
                 f"the context does not contain the answer, politely state that you cannot find the information. Do "
                 f"not make up any information. Context: {context} Question: {request.question} Answer:"
    )
    prompt = prompt_template.format(context=context, question=request.question)

    # Generate answer using LLM
    try:
        answer = llm.invoke(prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {exc}")

    return QueryResponse(answer=answer.strip(), sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000))
    )
