import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("📚 Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDF and chat with the content.")

# API Key input
api_key = st.text_input("Enter your Groq API key", type="password")

if api_key:
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model='gemma2-9b-it')

    # Session input
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize session state
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF file(s)", type='pdf', accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = './temp_' + uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
            os.remove(temp_path)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Vectorstore creation using new Chroma client
        client = chromadb.PersistentClient(path="./chroma_db")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client=client,
            collection_name="pdf_rag_docs"
        )
        retriever = vectorstore.as_retriever()

        # Prompt templates
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question "
                       "which might reference context in the chat history, "
                       "formulate a standalone question which can be understood "
                       "without the chat history. Do NOT answer the question, "
                       "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # RAG Chain
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # History handler
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat interface
        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.subheader("🧠 Assistant:")
            st.write(response.get("answer", "No response."))

            st.subheader("📜 Chat History:")
            for msg in session_history.messages:
                st.write(f"**{msg.type.capitalize()}**: {msg.content}")
else:
    st.warning("Please enter your API key to continue.")
