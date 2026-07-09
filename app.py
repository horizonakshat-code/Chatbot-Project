import os
import tempfile

import streamlit as st
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

st.set_page_config(
    page_title="Technical Documentation RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Technical Documentation RAG Chatbot")
st.markdown("Upload an HTML document and chat with it.")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload HTML File",
    type=["html", "htm"]
)

if uploaded_file and api_key:

    os.environ["OPENAI_API_KEY"] = api_key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(uploaded_file.read())
        html_path = tmp.name

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    question = st.chat_input("Ask something about the document")

    if question:

        with st.chat_message("user"):
            st.write(question)

        docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        with st.chat_message("assistant"):
            st.write(response.content)

else:
    st.info("Upload an HTML file and enter your OpenAI API key.")
