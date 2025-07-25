# Simple Multilingual RAG System

This project implements a basic Retrieval-Augmented Generation (RAG) system capable of understanding and responding to queries in both English and Bengali. The system uses a Bengali textbook as its knowledge base and exposes its functionality through a simple REST API.

## Features
- **Multilingual Support:** Handles queries in both English and Bengali.
- **Advanced Text Extraction:** Uses an OCR-based pipeline with image pre-processing to extract text from scanned PDFs.
- **Robust Retrieval:** Implements a "retrieve-then-rerank" strategy for improved accuracy.
- **REST API:** Provides a simple `FastAPI` endpoint for easy interaction.
- **Local First:** Uses FAISS for local vector storage, requiring no external database setup.

## Tech Stack
- **Backend/API:** FastAPI, Uvicorn
- **Core Logic:** LangChain
- **Vector Database:** FAISS (local)
- **Text Extraction:** PyTesseract, OpenCV, pdf2image
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Re-ranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Google Gemini (`gemini-1.5-flash`)

## Setup Guide

### 1. System Dependencies
This project relies on Tesseract for OCR. You must install it and its Bengali language pack.

**On Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ben poppler-utils
```
**On macOS (using Homebrew):**
```bash
brew install tesseract tesseract-lang poppler
```

### 2. Project Setup
1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Key:**
    - Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Create a file named `.env` in the project's root directory.
    - Add your API key to the `.env` file:
      ```
      GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
      ```

5.  **Add Knowledge Base:**
    - Place your PDF file named `HSC26-Bangla-1st-Paper.pdf` in the root directory of the project.

## How to Run

The application has two main commands.

### Step 1: Build the Knowledge Base
First, you need to process the PDF and create the vector database. This is a one-time process.

```bash
python main.py build-db
```
This will create a folder named `faiss_index` in your directory.

### Step 2: Run the API Server
Once the knowledge base is built, you can start the API server.

```bash
python main.py run-api
```
The server will start on `http://localhost:8000`.

## API Documentation

### Chat Endpoint
- **URL:** `/api/chat`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "query": "আপনার প্রশ্ন এখানে লিখুন"
  }
  ```
- **Example with `curl`:**
  ```bash
  curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"}'
  ```
- **Success Response (200 OK):**
  ```json
  {
    "answer": "প্রদত্ত তথ্য অনুযায়ী, বিয়ের সময় কল্যাণীর বয়স ছিল পনেরো বছর।"
  }
  ```

## Sample Queries and Outputs

### Bengali
- **Question:** `অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`
- **Expected Answer:** `শম্ভুনাথবাবুকে`

- **Question:** `কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?`
- **Expected Answer:** `মামাকে`

### English
- **Question:** `What was the first chatbot?`
- **Expected Answer:** Eliza`

- **Question:** `When did AI research formally begin?`
- **Expected Answer:** `AI research formally began in 1956 at the Dartmouth Conference.`
