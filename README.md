# Local Multimodal Research Agent

A privacy-first, fully local Retrieval-Augmented Generation (RAG) AI assistant designed to intelligently analyze and compare complex documents. Unlike standard RAG systems that discard visual data, this agent natively understands **text, tables, charts, and handwritten notes** simultaneously.

## Key Features
* **100% Local & Private**: Powered by locally hosted LLMs via Ollama. No data is ever sent to external APIs.
* **Multimodal Vision-Language processing**: Uses `Qwen2.5-VL` to process and understand visual elements (figures, diagrams) directly from PDF pages alongside the extracted text.
* **Hybrid Search Retrieval**: Combines semantic vector search (FAISS + Sentence-Transformers) with exact keyword matching (BM25) and Reciprocal Rank Fusion for high-accuracy evidence retrieval.
* **Multilingual Hybrid OCR**: Integrates PaddleOCR (customized for Hindi/Devanagari, Kannada, and Latin scripts) and Tesseract to read text perfectly, even from scanned images.
* **Multi-Document Comparison**: Chat seamlessly across multiple uploaded PDFs simultaneously without data bleeding.
* **Sleek UI**: Modern, glassmorphism-inspired React/Vite frontend with drag-and-drop support, inline markdown rendering, and exact page-level source citations.

## Tech Stack
* **Frontend**: React, Vite, Vanilla CSS
* **Backend**: FastAPI (Python), Uvicorn
* **AI & Embedding**: Ollama (`qwen2.5-vl`), PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Retrieval & DB**: FAISS (in-memory), Rank-BM25
* **Document Processing**: PyMuPDF, PaddleOCR, Tesseract

## Installation & Running
1. Clone the repository and navigate into it.
2. Create and activate a Python virtual environment (`python -m venv .venv`).
3. Install dependencies: `pip install -r requirements.txt`.
4. Install and start [Ollama](https://ollama.com/), then run `ollama run qwen2.5-vl`.
5. Run the application: Execute `start.bat` (Windows) to automatically launch both the FastAPI backend and the React frontend.
