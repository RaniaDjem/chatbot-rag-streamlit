# utils.py
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import requests
import types
import streamlit as st


API_KEY = st.secrets["MISTRAL_API_KEY"]

def process_pdf(pdf_path):
    # Lecture et extraction du texte
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    # Découpage en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)

    # Embedding + vectorisation avec FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    return vectorstore


def query_mistral(prompt, model="mistral-small"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_chain(vectorstore):
    prompt_template = "Réponds à la question suivante basée sur le contexte ci-dessous :\n\n{context}\n\nQuestion : {question}"

    def chain(question):
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = prompt_template.format(context=context, question=question)
        return query_mistral(prompt)

    return types.SimpleNamespace(run=chain)
