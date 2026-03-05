# KSelect UI Test

Drag-and-drop CSV chatbot powered by KSelect + Claude.

## Quick start

### 1. Backend (FastAPI)

```bash
# From project root — uses the existing .venv
cd /Users/jhonathanherrera/KSelect/KSelect

# Install extra deps into the existing venv
.venv/bin/pip install fastapi "uvicorn[standard]" python-multipart

# Run the API server
PYTHONPATH=. .venv/bin/uvicorn UI_Test.backend.api:app --reload --port 8000
```

Make sure `ANTHROPIC_API_KEY` is in your `.env` at the project root.

### 2. Frontend (React + Vite)

```bash
cd UI_Test/frontend
npm install
npm run dev
```

Open http://localhost:5173

## What it does

1. Drop a CSV file → pick which column is the text to index
2. KSelect builds a FAISS + BM25 index in the background
3. Type questions in the chat — Claude answers using retrieved chunks
4. Each answer shows:
   - **Confidence bar** (0–100%)
   - **Retrieval time** / **LLM time** / **Total time**
   - **Chunks retrieved / in-context / dropped**
   - **Context tokens used**
   - Expandable **sources** with individual relevance scores
5. Left sidebar shows index info (filename, column list, chunk count, build time)
