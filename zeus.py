#!/usr/bin/env python3

import os
import glob
import numpy as np
import PyPDF2
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from fuzzywuzzy import process
import streamlit as st

########################################################################
# Utility Functions
########################################################################

def chunk_text(text, chunk_size=100):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i: i + chunk_size])

def read_pdf(pdf_path):
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

########################################################################
# Index-Building Functions
########################################################################

def build_faiss_index(pdf_dir, chunk_size=100, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        return None, None, None
    
    chunked_corpus = []
    doc_map = []
    for pdf_path in pdf_files:
        pdf_text = read_pdf(pdf_path)
        for chunk in chunk_text(pdf_text, chunk_size=chunk_size):
            cleaned_chunk = chunk.strip().replace("\n", " ")
            if cleaned_chunk:
                chunked_corpus.append(cleaned_chunk)
                doc_map.append(pdf_path)

    embedder = SentenceTransformer(embedding_model_name)
    embeddings = embedder.encode(chunked_corpus, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunked_corpus, doc_map

########################################################################
# Query Functions
########################################################################

def retrieve_relevant_chunks(query, index, chunked_corpus, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=3):
    embedder = SentenceTransformer(embedding_model_name)
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunked_corpus[i] for i in indices[0] if i < len(chunked_corpus)]
    return relevant_chunks

def load_generator(model_name="EleutherAI/gpt-neo-1.3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_answer(query, retrieved_chunks, tokenizer, model, max_length=256, temperature=0.7, top_p=0.9):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature, top_p=top_p, do_sample=True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def simple_response(query, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

########################################################################
# Streamlit App
########################################################################

def main():
    st.title("Interactive RAG System")
    st.sidebar.title("Options")
    st.sidebar.subheader("Upload PDFs")
    
    # Use session state to persist data
    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.chunked_corpus = None
        st.session_state.doc_map = None

    # Upload and process PDFs
    pdf_dir = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if pdf_dir:
        pdf_dir_path = "./uploaded_pdfs"
        os.makedirs(pdf_dir_path, exist_ok=True)
        for pdf in pdf_dir:
            with open(os.path.join(pdf_dir_path, pdf.name), "wb") as f:
                f.write(pdf.getbuffer())
        st.sidebar.success("PDFs uploaded successfully.")

        if st.sidebar.button("Build FAISS Index"):
            st.session_state.index, st.session_state.chunked_corpus, st.session_state.doc_map = build_faiss_index(pdf_dir_path)
            if st.session_state.index:
                st.sidebar.success("Index built successfully!")
            else:
                st.sidebar.error("No PDFs found to process.")

    # Query system
    user_query = st.text_input("Ask a question:")
    if st.button("Submit"):
        if st.session_state.index and st.session_state.chunked_corpus:
            relevant_chunks = retrieve_relevant_chunks(user_query, st.session_state.index, st.session_state.chunked_corpus)
            if relevant_chunks:
                tokenizer, model = load_generator()
                answer = generate_answer(user_query, relevant_chunks, tokenizer, model)
                st.write(f"**Answer:** {answer}")
            else:
                st.warning("No relevant information found. Using lightweight model.")
                simple_answer = simple_response(user_query)
                st.write(f"**Answer:** {simple_answer}")
        else:
            st.warning("No resources indexed. Using lightweight model.")
            simple_answer = simple_response(user_query)
            st.write(f"**Answer:** {simple_answer}")

if __name__ == "__main__":
    main()
