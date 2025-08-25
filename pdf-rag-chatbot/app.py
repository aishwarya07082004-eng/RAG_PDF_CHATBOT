import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import re
from typing import List, Dict, Tuple

# -------------------------
# Embedding + Retrieval
# -------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"  # lightweight, CPU-friendly text2text model

_embedder = None
_tokenizer = None
_lm = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder

def get_llm():
    global _lm, _tokenizer
    if _lm is None:
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _lm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    return _lm, _tokenizer

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    # Simple word-based chunking
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def extract_pdf_chunks(pdf_path: str, doc_label: str, chunk_size: int = 500, overlap: int = 100):
    reader = PdfReader(pdf_path)
    all_chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        for ch in chunk_text(text, chunk_size, overlap):
            all_chunks.append({
                "text": ch,
                "doc": doc_label,
                "page": page_num
            })
    return all_chunks

def build_index(file_objs) -> Tuple[Dict, str]:
    """
    file_objs: list of tempfile objects from Gradio (len up to 2)
    Returns (index, status_msg)
    """
    if not file_objs or len(file_objs) == 0:
        return {}, "Please upload up to two PDFs."
    if len(file_objs) > 2:
        file_objs = file_objs[:2]
    embedder = get_embedder()
    corpus = []
    for i, f in enumerate(file_objs, start=1):
        label = f"Doc{i}: {os.path.basename(f.name)}"
        chunks = extract_pdf_chunks(f.name, label)
        corpus.extend(chunks)
    if not corpus:
        return {}, "No extractable text found in the PDFs."
    texts = [c["text"] for c in corpus]
    embs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    index = {
        "corpus": corpus,
        "embs": embs.astype(np.float32)  # save memory
    }
    return index, f"Indexed {len(corpus)} chunks from {len(file_objs)} file(s)."

def retrieve(query: str, index: Dict, k: int = 5):
    if not query or not index:
        return []
    embedder = get_embedder()
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = index["embs"] @ q  # cosine because both normalized
    topk_idx = np.argsort(-scores)[:k]
    results = []
    for rank, idx in enumerate(topk_idx, start=1):
        item = index["corpus"][idx]
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "text": item["text"],
            "doc": item["doc"],
            "page": item["page"]
        })
    return results

def make_prompt(query: str, contexts: List[Dict]) -> str:
    context_block = "\n\n".join(
        [f"[{c['doc']} | p.{c['page']}] {c['text']}" for c in contexts]
    )
    sys = (
        "You are a helpful assistant. Answer the user's question ONLY using the context. "
        "If the answer is not found, say you don't know. Provide citations like (DocX p.Y) after each fact."
    )
    prompt = (
        f"{sys}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt

def answer_question(query: str, index: Dict, max_new_tokens: int = 256, temperature: float = 0.3):
    if not index:
        return "Please upload PDFs and click 'Build Index' first.", None
    ctx = retrieve(query, index, k=6)
    if not ctx:
        return "I couldn't find relevant content in the PDFs.", None
    prompt = make_prompt(query, ctx)
    model, tok = get_llm()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1
        )
    text = tok.decode(outputs[0], skip_special_tokens=True)

    # Build sources table
    src_lines = []
    for c in ctx:
        src_lines.append(f"{c['rank']}. {c['doc']} (p.{c['page']}) — score: {c['score']:.3f}")
    sources = "\n".join(src_lines)
    return text, sources

# -------------------------
# Gradio UI
# -------------------------

index_state = gr.State({})

with gr.Blocks(title="Simple PDF RAG Chatbot") as demo:
    gr.Markdown(
        """
        # 🧠 Simple PDF RAG Chatbot
        Upload up to **two PDFs**, build the index, and ask questions grounded in those documents.
        """
    )
    with gr.Row():
        files = gr.File(label="Upload PDFs (max 2)", file_count="multiple", file_types=[".pdf"])
    build_btn = gr.Button("🔧 Build Index")
    status = gr.Markdown("")
    with gr.Row():
        query = gr.Textbox(label="Ask a question", placeholder="e.g., What is the main conclusion in Doc 1?")
    ask_btn = gr.Button("💬 Ask")
    answer = gr.Markdown(label="Answer")
    sources = gr.Textbox(label="Retrieved Chunks (top-k)", lines=8)

    def _build(files_list):
        try:
            idx, msg = build_index(files_list or [])
            return idx, msg
        except Exception as e:
            return {}, f"Error while indexing: {e}"

    def _ask(q, idx):
        try:
            text, src = answer_question(q, idx)
            return text, src or ""
        except Exception as e:
            return f"Error while answering: {e}", ""

    build_btn.click(_build, inputs=[files], outputs=[index_state, status])
    ask_btn.click(_ask, inputs=[query, index_state], outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()