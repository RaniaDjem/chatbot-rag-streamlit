import streamlit as st
import sys
import tempfile
from utils import process_pdf, get_chain

# Ignore the torch module from being watched by Streamlit to avoid path issues
sys.modules["torch.classes"] = None

st.set_page_config(page_title="Chatbot RAG PDF", page_icon="🧠")

# 📝 Introduction
st.title("📄 Chatbot Intelligent sur Document PDF")
st.markdown("""
Bienvenue sur **Chatbot RAG PDF** ! 🚀

Téléversez simplement un fichier **PDF** (article, rapport, cours, etc.), et posez **toutes vos questions** dessus ✨  
Notre assistant intelligent lit le document et vous répond précisément, en s'appuyant sur le contenu du PDF.

_C’est comme avoir un assistant personnel qui lit pour vous et vous résume tout_ 🤖📄
""")

# 📎 Fichier PDF
uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("Document chargé. Initialisation en cours...")
    vectorstore = process_pdf(pdf_path)
    qa_chain = get_chain(vectorstore)

    st.success("Chatbot prêt ! Posez votre question.")
    question = st.text_input("Votre question :")

    if question:
        with st.spinner("Génération de la réponse..."):
            response = qa_chain.run(question)
            st.markdown(f"**Réponse :** {response}")

# 🛠️ Stack technique & crédits
st.markdown("""
---
🛠️ **Stack technique** utilisée :
- [Streamlit](https://streamlit.io/) pour l'interface web
- [LangChain](https://www.langchain.com/) pour orchestrer le RAG (Retrieval-Augmented Generation)
- [Mistral AI](https://mistral.ai/) pour le modèle de langage (LLM)
- [Hugging Face Transformers](https://huggingface.co/) pour les embeddings
- [FAISS](https://github.com/facebookresearch/faiss) pour la recherche sémantique

💡 Ce projet a été réalisé par **Rania Djema**.
""")
