import streamlit as st
import sys

# Ignore the torch module from being watched by Streamlit to avoid path issues
sys.modules["torch.classes"] = None

import tempfile
from utils import process_pdf, get_chain
import types

# app.py

st.set_page_config(page_title="Chatbot RAG PDF", page_icon="🧠")

st.title("📄 Chatbot Intelligent sur Document PDF")
st.markdown("Posez vos questions sur un document PDF avec un modèle **LLM Mistral + RAG**.")

uploaded_file = st.file_uploader("📎 Choisissez un fichier PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ Document chargé. Initialisation en cours...")
    vectorstore = process_pdf(pdf_path)
    qa_chain = get_chain(vectorstore)

    st.success("✅ Chatbot prêt ! Posez votre question.")
    question = st.text_input("💬 Votre question :")

    if question:
        with st.spinner("⏳ Génération de la réponse..."):
            try:
                response = qa_chain.run(question)
                st.markdown(f"**📢 Réponse :** {response}")
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
