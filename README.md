# 🧠 DocMind AI — Complete Setup & Architecture Guide

> A fully local RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDF documents and ask questions about them using natural language. Powered by Llama 3.1 via Groq, ChromaDB, and Gradio.

---

## 📋 Table of Contents

1. [What is This?](#what-is-this)
2. [Architecture Overview](#architecture-overview)
3. [Tech Stack](#tech-stack)
4. [How RAG Works](#how-rag-works)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [Running the App](#running-the-app)
7. [How to Use the App](#how-to-use-the-app)
8. [Deep Dive: Each Component](#deep-dive-each-component)
9. [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)

---

## What is This?

DocMind AI is a **Retrieval-Augmented Generation (RAG)** chatbot. Instead of relying on an AI model's pre-trained knowledge, it reads your PDF documents, stores their content in a vector database, and retrieves the most relevant passages every time you ask a question — passing them to an LLM to generate a grounded, accurate answer.

**Key properties:**
- ✅ Answers are always grounded in your documents — no hallucination
- ✅ Works with multiple PDFs at once
- ✅ Completely free (Groq free tier + local embeddings)
- ✅ No data leaves your machine except the final LLM call
- ✅ Runs on any Windows/Mac/Linux machine

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    Gradio Web App (port 7860)                   │
└────────────────────────┬───────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │        Two Phases           │
          └──────────────┬──────────────┘
                         │
     ┌───────────────────┼────────────────────┐
     │                                        │
     ▼                                        ▼
┌─────────────────┐                  ┌─────────────────┐
│  INDEXING PHASE │                  │  QUERYING PHASE │
│  (on upload)    │                  │  (on question)  │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────┐                  ┌─────────────────┐
│   PDF Loader    │                  │  User Question  │
│ (PyPDFLoader)   │                  └────────┬────────┘
└────────┬────────┘                           │
         │                                    ▼
         ▼                          ┌──────────────────┐
┌─────────────────┐                 │  Embed Question  │
│  Text Splitter  │                 │  (MiniLM-L6-v2)  │
│ (chunk: 600     │                 └────────┬─────────┘
│  overlap: 150)  │                          │
└────────┬────────┘                          ▼
         │                          ┌──────────────────┐
         ▼                          │  Vector Search   │
┌─────────────────┐                 │  in ChromaDB     │
│   Embeddings    │                 │  (top 4 chunks)  │
│  (MiniLM-L6-v2) │                 └────────┬─────────┘
│  384 dimensions │                          │
└────────┬────────┘                          ▼
         │                          ┌──────────────────┐
         ▼                          │  Build Prompt    │
┌─────────────────┐                 │  context +       │
│    ChromaDB     │                 │  question        │
│  Vector Store   │                 └────────┬─────────┘
│  (persisted     │                          │
│   on disk)      │                          ▼
└─────────────────┘                 ┌──────────────────┐
                                    │   Groq API       │
                                    │ Llama-3.1-8B     │
                                    │ (LLM inference)  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  Final Answer    │
                                    │  displayed in    │
                                    │  chat UI         │
                                    └──────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **UI** | Gradio 6.15 | Web interface for chat and file upload |
| **PDF Parsing** | LangChain + PyPDF | Extract text from uploaded PDFs |
| **Text Splitting** | RecursiveCharacterTextSplitter | Break documents into overlapping chunks |
| **Embedding Model** | `all-MiniLM-L6-v2` (HuggingFace) | Convert text to 384-dim vectors — runs locally |
| **Vector Database** | ChromaDB | Store and search embeddings on disk |
| **LLM** | Llama 3.1-8B-Instant | Generate answers — called via Groq API |
| **LLM Client** | OpenAI Python SDK | HTTP client (pointed at Groq's endpoint) |

---

## How RAG Works

RAG has two distinct phases:

### Phase 1 — Indexing (happens when you upload PDFs)

```
PDF File(s)
    │
    ▼
Extract raw text from every page
    │
    ▼
Split into overlapping chunks
  • chunk_size  = 600 characters
  • chunk_overlap = 150 characters
  (overlap ensures context isn't lost at chunk boundaries)
    │
    ▼
Embed each chunk using MiniLM-L6-v2
  • Each chunk → 384-dimensional float vector
  • Captures semantic meaning, not just keywords
    │
    ▼
Store all vectors in ChromaDB (saved to ./chroma_db on disk)
```

### Phase 2 — Querying (happens every time you ask a question)

```
User types a question
    │
    ▼
Embed the question using the SAME MiniLM-L6-v2 model
  • Question → 384-dimensional float vector
    │
    ▼
Cosine similarity search in ChromaDB
  • Compare question vector against all stored chunk vectors
  • Retrieve top K=4 most similar chunks
    │
    ▼
Build a prompt:
  [SYSTEM]: You are DocMind. Answer using ONLY the context below...
  [CONTEXT]: chunk1 --- chunk2 --- chunk3 --- chunk4
  [QUESTION]: user's question
    │
    ▼
Send prompt to Groq API → Llama 3.1-8B-Instant
    │
    ▼
Return answer to the chat UI
```

---

## Step-by-Step Setup

### Prerequisites

- Python 3.10 or higher
- Windows / macOS / Linux
- Internet connection (for first run — to download embedding model)
- A free Groq API key

---

### Step 1 — Get a Free Groq API Key

1. Go to **https://console.groq.com**
2. Sign up for a free account (no credit card needed)
3. Go to **https://console.groq.com/keys**
4. Click **"Create API Key"**
5. Give it a name (e.g. `docmind`)
6. Copy the key — it starts with `gsk_` and is only shown once
7. Save it somewhere safe

---

### Step 2 — Create a Virtual Environment

Open a terminal/PowerShell in your project folder:

```powershell
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- `gradio` — the web UI framework
- `openai` — HTTP client for Groq API calls
- `langchain-text-splitters` — document chunking
- `langchain-community` — PDF loader
- `langchain-chroma` — ChromaDB integration
- `langchain-huggingface` — local embedding model
- `chromadb` — vector database
- `sentence-transformers` — runs MiniLM-L6-v2 locally
- `pypdf` — PDF text extraction
- `torch` — required by sentence-transformers

> ⏱️ First install takes 5–10 minutes. PyTorch is large (~200MB).

---

### Step 4 — Run the App

```powershell
python app.py
```

The first time you run it, it will download the embedding model
(`all-MiniLM-L6-v2`, ~90MB) from HuggingFace. This only happens once —
it's cached locally after that.

Your browser will open automatically at:
```
http://127.0.0.1:7860
```

---

## Running the App

```powershell
# Make sure venv is active
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Start the app
python app.py
```

To stop: press `Ctrl+C` in the terminal.

If port 7860 is already in use:
```powershell
# Find and kill the process using port 7860
netstat -ano | findstr :7860
taskkill /PID <PID_NUMBER> /F

# Or just change the port in app.py:
# server_port=7861
```

---

## How to Use the App

1. **Paste your Groq API key** into the "Groq API Key" field in the left sidebar
2. **Upload your PDF(s)** — drag and drop or click to browse. Multiple files supported
3. **Click "🚀 Process Documents"** — wait for the status box to show ✅ Ready
4. **Type your question** in the chat box at the bottom and press Enter or click Send
5. **Read the answer** — DocMind will respond based only on your documents
6. **Click "🗑️ Clear Session"** to reset and start fresh with new documents

---

## Deep Dive: Each Component

### 1. PDF Loading — `PyPDFLoader`

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages  = loader.load()
# Returns a list of Document objects, one per page
# Each Document has .page_content (text) and .metadata (page number, source)
```

PyPDFLoader uses the `pypdf` library under the hood to extract raw text
from each page of the PDF. It handles multi-page documents automatically.

---

### 2. Text Splitting — `RecursiveCharacterTextSplitter`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,     # max characters per chunk
    chunk_overlap=150,  # characters shared between adjacent chunks
)
chunks = splitter.split_documents(pages)
```

**Why split?** LLMs have a context window limit. A 100-page PDF might be
200,000 characters — way too large to send at once. Splitting breaks it
into manageable pieces.

**Why overlap?** If a sentence spans a chunk boundary, overlap ensures
both chunks contain enough context to make sense independently.

**Recursive strategy:** It tries to split on paragraphs first, then
sentences, then words, then characters — always trying to keep natural
language boundaries intact.

---

### 3. Embedding — `all-MiniLM-L6-v2`

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

**What is an embedding?**
An embedding is a list of 384 floating point numbers (a vector) that
represents the semantic meaning of a piece of text. Texts with similar
meanings have vectors that are close together in this 384-dimensional space.

**Why MiniLM-L6-v2?**
- Runs entirely on your CPU — no GPU needed
- Only 90MB in size
- Very fast — embeds hundreds of chunks in seconds
- Good accuracy for semantic similarity tasks
- Free, open source, no API calls needed

**normalize_embeddings=True** ensures all vectors have length 1, which
makes cosine similarity equivalent to dot product — faster computation.

---

### 4. Vector Database — ChromaDB

```python
from langchain_chroma import Chroma

# Store embeddings
vectordb = Chroma.from_documents(
    documents=chunks,         # your text chunks
    embedding=embedding,      # the embedding function
    persist_directory="./chroma_db",  # save to disk
)

# Search for similar chunks
docs = vectordb.similarity_search(question, k=4)
```

**What is ChromaDB?**
ChromaDB is an open-source vector database that stores embeddings and
performs fast similarity searches. It runs as an in-process library —
no separate server needed.

**How similarity search works:**
1. Embed the user's question → 384-dim vector
2. Compare against every stored chunk vector using cosine similarity
3. Cosine similarity = dot product of two normalized vectors
4. Higher score = more semantically similar
5. Return top K=4 chunks with highest similarity scores

**Cosine similarity formula:**
```
similarity = (A · B) / (|A| × |B|)
```
Where A is the question vector and B is a chunk vector.
Result is between -1 and 1. Closer to 1 = more similar.

**Persistence:** ChromaDB saves its index to `./chroma_db/` on disk.
Each time you upload new documents, the old DB is deleted and rebuilt fresh.

---

### 5. Prompt Construction

```python
SYSTEM_PROMPT = (
    "You are DocMind, a brilliant and precise AI assistant. "
    "Answer using ONLY the document context provided. "
    "Never fabricate beyond the given context."
)

messages = [
    {"role": "system",  "content": SYSTEM_PROMPT},
    {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {question}"},
]
```

The retrieved chunks are joined with `---` separators and injected into
the user message as context. The system prompt instructs the LLM to stay
grounded and not hallucinate beyond what's in the context.

---

### 6. LLM Inference — Llama 3.1-8B via Groq

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    max_tokens=600,
    temperature=0.3,
)

answer = response.choices[0].message.content.strip()
```

**Why Groq?**
Groq runs LLMs on custom LPU (Language Processing Unit) hardware that
is significantly faster than traditional GPU inference. Their free tier
gives you ~14,400 requests/day — more than enough for personal use.

**Why OpenAI SDK?**
Groq's API is fully OpenAI-compatible. Using the OpenAI Python SDK
pointed at Groq's base URL avoids provider routing issues entirely.

**temperature=0.3** keeps responses focused and consistent.
Lower temperature = more deterministic. Higher = more creative.

**max_tokens=600** limits response length to keep answers concise.

---

### 7. Gradio UI

```python
import gradio as gr

with gr.Blocks() as demo:
    chatbot  = gr.Chatbot(...)   # chat history display
    msg      = gr.Textbox(...)   # user input
    send_btn = gr.Button(...)    # send button

    send_btn.click(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, msg])

demo.launch(server_name="127.0.0.1", server_port=7860)
```

Gradio runs a FastAPI server locally and serves a React frontend.
The `gr.Blocks` API lets you build custom layouts with full control
over component placement and styling.

---

## Project Structure

```
chatbot/
│
├── app.py              # Main application — all logic and UI
├── requirements.txt    # Python dependencies
├── README.md           # This file
│
└── chroma_db/          # Auto-created when you process documents
    ├── chroma.sqlite3  # ChromaDB metadata and index
    └── ...             # Vector data files
```

The `chroma_db/` folder is created automatically when you first process
documents. It is deleted and rebuilt every time you click
"Process Documents" to ensure a clean index.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Package not installed | Run `pip install -r requirements.txt` |
| `401 Invalid API Key` | Wrong Groq key | Generate a new key at console.groq.com |
| `Port 7860 in use` | Previous run still open | Kill the process or change `server_port` to 7861 |
| `No module langchain.text_splitter` | Old LangChain import | Use `langchain_text_splitters` instead |
| App opens but chat does nothing | Documents not processed | Click "Process Documents" first |
| Slow first run | Downloading embedding model | Wait ~1 min — only happens once |
| `chroma_db` error | Corrupted vector store | Delete the `chroma_db/` folder and reprocess |

---

## Free Tier Limits

| Service | Free Limit | Notes |
|---|---|---|
| **Groq** | ~14,400 req/day | Resets every 24 hours |
| **HuggingFace embeddings** | Unlimited | Runs locally, no API calls |
| **ChromaDB** | Unlimited | Runs locally on disk |

For personal and small-team use, the Groq free tier is more than sufficient.

---

*Built with ❤️ using LangChain · ChromaDB · Groq · Gradio*
