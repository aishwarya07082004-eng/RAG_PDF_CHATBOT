import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global FAISS index and docs
index = None
documents = []

def build_index(pdf1, pdf2):
    global index, documents
    documents = []

    # Read PDFs
    for pdf in [pdf1, pdf2]:
        if pdf is not None:
            reader = PdfReader(pdf.name)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    documents.append(text)

    # Embed and store in FAISS
    embeddings = embedder.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return "Index built successfully! You can now ask questions."

def chatbot_response(message, history):
    global index, documents
    if index is None:
        history.append((message, "⚠️ Please upload PDFs and build the index first."))
        return history

    # Embed query
    query_vec = embedder.encode([message])
    D, I = index.search(np.array(query_vec), k=2)

    # Retrieve top docs
    context = "\n".join([documents[i] for i in I[0]])
    response = f"Answer based on docs:\n\n{context[:1000]}"

    # Append new interaction
    history.append((message, response))
    return history

with gr.Blocks() as demo:
    gr.Markdown("## 📘 PDF-based RAG Chatbot")

    with gr.Row():
        pdf1 = gr.File(label="Upload PDF 1", file_types=[".pdf"])
        pdf2 = gr.File(label="Upload PDF 2", file_types=[".pdf"])

    build_btn = gr.Button("Build Index")
    status = gr.Textbox(label="Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your question")
    clear = gr.Button("Clear")

    build_btn.click(build_index, inputs=[pdf1, pdf2], outputs=status)
    msg.submit(chatbot_response, inputs=[msg, chatbot], outputs=chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)  # clear chat

demo.launch()
