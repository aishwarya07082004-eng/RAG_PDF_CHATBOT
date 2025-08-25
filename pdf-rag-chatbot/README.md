# Simple PDF-based RAG Chatbot (Gradio + HF Spaces)

A minimal Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in **up to two uploaded PDF files**.  
Runs fully on CPU using:
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generator: `google/flan-t5-base`
- UI: Gradio

## ✨ Features
- Upload **two PDF documents**.
- Automatic text extraction + chunking with page tracking.
- Fast cosine-similarity retrieval over sentence embeddings.
- Answers strictly from context; includes simple citations like `(DocX p.Y)`.
- One-file `app.py` deployable on **Hugging Face Spaces** and locally.

## 🚀 Quickstart (Local)
```bash
# 1) Clone and enter
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# 2) Create venv (recommended)
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate

# 3) Install
pip install -r requirements.txt

# 4) Run
python app.py
# Open the Gradio link printed in the terminal
```

## 🌐 Deploy to Hugging Face Spaces
1. Create a new **Space**: **New Space → Gradio → Public** (hardware: CPU Basic is OK).
2. Upload these three files:
   - `app.py`
   - `requirements.txt`
   - `README.md` (optional)
3. The Space will auto-build and launch your app.

> Tip: You can also push the repo with `git`:
```bash
# After creating an empty Space named your-username/pdf-rag-chatbot
git init
git remote add origin https://huggingface.co/spaces/your-username/pdf-rag-chatbot
git add .
git commit -m "Initial commit: simple PDF RAG chatbot"
git push origin main
```

## 🧩 How it Works
1. **Indexing**  
   - Extracts text per page with `pypdf`  
   - Chunks into ~500-word windows with 100-word overlap  
   - Embeds chunks via `all-MiniLM-L6-v2` (L2-normalized)  
2. **Retrieval**  
   - Cosine similarity between query and chunk embeddings  
   - Top-k (k=6) chunks become the **context**  
3. **Generation**  
   - Builds a compact prompt instructing the model to only use provided context  
   - Uses `flan-t5-base` for grounded answers (CPU-friendly)

## 🧪 Notes
- If your PDFs are scanned images, text extraction may be empty (no OCR). Use OCRed PDFs.
- You can switch the generator model by setting `LLM_MODEL_NAME` in `app.py` to something like
  - `google/flan-t5-large` (better, slower)  
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (change code to text-generation pipeline)  
- For larger corpora, consider FAISS/Chroma; here we keep it minimal and dependency-light.

## 📄 License
MIT