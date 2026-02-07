# YKS AI RAG

A RAG (Retrieval Augmented Generation) based assistant for question solving and topic explanation, developed for YKS (Higher Education Institutions Examination) students.

## Features

- **Local Document Processing**: Processes lecture notes and books in PDF and TXT formats.
- **Vector Search**: Finds relevant content, formulas, and sample questions quickly using FAISS.
- **Smart Solution**: Generates step-by-step, understandable mathematical solutions using Google Gemini API (Gemini 2.5).
- **Fast and Lightweight**: Does not require a GPU, can run on CPU.

## Installation

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Copy `.env.example` to `.env` and add your Gemini API key.
   ```bash
   copy .env.example .env
   ```
   Open `.env` and enter your `GEMINI_API_KEY`.
   
   It is also recommended to add `HF_TOKEN` (Hugging Face Token) (https://huggingface.co/settings/tokens):
   ```
   HF_TOKEN=hf_...
   ```

## Usage

### 1. Adding and Indexing Documents
Place your PDF or TXT files in the `documents/` folder. A sample file `konu_anlatimi_ornek.txt` is included.

To start indexing:
```bash
python -m ingest.ingest_documents
```
This process splits documents into chunks, generates embeddings, and saves them to the `index/` folder.

### 2. Starting the Backend
Start the API server:
```bash
uvicorn app.main:app --reload
```
The server will run at `http://localhost:8000`.

### 3. Asking Questions
You can ask questions by sending a POST request while the API is running.

**Example Request (Curl):**
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What are the roots of x^2 + 5x + 6 = 0?\"}"
```

**Example Request (Python):**
```python
import requests

url = "http://localhost:8000/ask"
payload = {"question": "What are the roots of x^2 + 5x + 6 = 0?"}
response = requests.post(url, json=payload)
print(response.json())
```

## Project Structure
- `app/`: Main application code (API, Core, Utils)
- `ingest/`: Document processing scripts
- `documents/`: Source documents
- `index/`: Vector database files
