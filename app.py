import os
import tempfile

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

st.set_page_config(
    page_title="Technical Document RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Technical Documentation RAG Chatbot")
st.write("Upload an HTML document and ask questions.")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload HTML File",
    type=["html", "htm"]
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

if uploaded_file and api_key:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(uploaded_file.read())
        html_path = tmp.name

    loader = UnstructuredHTMLLoader(html_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    question = st.text_input("Ask your question")

    if st.button("Get Answer"):

        if question.strip():

            with st.spinner("Thinking..."):

                answer = qa.run(question)

            st.success("Answer")

            st.write(answer)

else:
    st.info("Upload an HTML file and enter your OpenAI API key.")
