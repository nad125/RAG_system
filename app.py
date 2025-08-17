# app.py

import os
import torch
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- CONFIGURATION ---
FAISS_INDEX_PATH = "faiss_index_txt"
EMBEDDING_MODEL_NAME = 'sentence-transformers/LaBSE'
LLM_MODEL_NAME = 'gemini-1.5-flash'
TEXT_FILE_PATH = "aparichita.txt"

# --- GLOBAL VARIABLES & INITIALIZATION ---
app = FastAPI()
rag_chain = None
text_chunks = []


# This function will be called on application startup
@app.on_event("startup")
def startup_event():
    global rag_chain, text_chunks

    # Check if the FAISS index exists; if not, build it.
    if not os.path.exists(FAISS_INDEX_PATH):
        print("INFO: FAISS index not found. Building it from text file...")
        with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
            text_content = f.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = [Document(page_content=text_content)]
        text_chunks = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.from_documents(text_chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("INFO: FAISS index built and saved.")
    else:
        print("INFO: Loading existing FAISS index...")
        # We need the chunks for BM25 even if FAISS is loaded
        with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
            text_content = f.read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = [Document(page_content=text_content)]
        text_chunks = text_splitter.split_documents(docs)
        print("INFO: Text chunks for BM25 loaded.")

    # Load all necessary components for the RAG chain
    print("INFO: Setting up RAG chain...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(text_chunks)
    bm25_retriever.k = 5

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.1)

    prompt_template = """
    You are a helpful assistant. Answer the user's question based ONLY on the following context.
    Your response must be in the same language as the user's question.
    If the context does not contain the answer, state that you cannot find the answer in the provided documents.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    rag_chain = (
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    print("✅ RAG Pipeline is ready.")


@app.get("/")
def read_root():
    return {"message": "RAG System is online!"}


@app.get("/query")
def get_answer(question: str):
    if not rag_chain:
        return {"error": "RAG chain not initialized. Please check server logs."}

    try:
        response = rag_chain.invoke(question)
        return {"question": question, "answer": response}
    except Exception as e:
        return {"error": f"An error occurred while processing the query: {e}"}