# app.py
import streamlit as st
import time
import base64
import os

from vectors import EmbeddingsManager
from chatbot import ChatbotManager


def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# Session state init
if "temp_pdf_path" not in st.session_state:
    st.session_state["temp_pdf_path"] = None
if "chatbot_manager" not in st.session_state:
    st.session_state["chatbot_manager"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.set_page_config(
    page_title="RAG Document Buddy",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## 📚 RAG Document Buddy")
    st.markdown("Chat with your PDFs — fully local, CPU-friendly.")
    st.markdown("---")
    menu = ["🏠 Home", "🤖 Chatbot"]
    choice = st.selectbox("Navigate", menu)

if choice == "🏠 Home":
    st.title("📄 RAG Document Buddy")
    st.markdown(
        """
        Welcome! This app runs **entirely locally**:
        - **BGE embeddings** (HuggingFace, CPU)
        - **Chroma** vector store (local folder, no server)
        - **Llama 3.2** via Ollama (local LLM)

        Upload a PDF, create embeddings, then ask questions about it.
        """
    )

elif choice == "🤖 Chatbot":
    st.title("🤖 Chat with your Document")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("📄 File Uploaded Successfully!")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")
            st.markdown("### 📖 PDF Preview")
            displayPDF(uploaded_file)

            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["temp_pdf_path"] = temp_pdf_path

    with col2:
        st.header("🧠 Embeddings")
        create_embeddings = st.checkbox("✅ Create Embeddings")
        if create_embeddings:
            if st.session_state["temp_pdf_path"] is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                try:
                    embeddings_manager = EmbeddingsManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        persist_dir="chroma_db",
                        collection_name="vector_db",
                    )
                    with st.spinner("🔄 Embeddings are in process..."):
                        result = embeddings_manager.create_embeddings(
                            st.session_state["temp_pdf_path"]
                        )
                        time.sleep(1)
                    st.success(result)

                    if st.session_state["chatbot_manager"] is None:
                        st.session_state["chatbot_manager"] = ChatbotManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_repo_id="Qwen/Qwen2.5-7B-Instruct",
                            llm_temperature=0.7,
                            persist_dir="chroma_db",
                            collection_name="vector_db",
                        )
                except FileNotFoundError as fnf_error:
                    st.error(fnf_error)
                except ValueError as val_error:
                    st.error(val_error)
                except ConnectionError as conn_error:
                    st.error(conn_error)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    with col3:
        st.header("💬 Chat with Document")
        if st.session_state["chatbot_manager"] is None:
            st.info("🤖 Please upload a PDF and create embeddings to start chatting.")
        else:
            for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).markdown(msg["content"])

            if user_input := st.chat_input("Type your message here..."):
                st.chat_message("user").markdown(user_input)
                st.session_state["messages"].append(
                    {"role": "user", "content": user_input}
                )

                with st.spinner("🤖 Responding..."):
                    try:
                        answer = st.session_state["chatbot_manager"].get_response(
                            user_input
                        )
                        time.sleep(1)
                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"

                st.chat_message("assistant").markdown(answer)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer}
                )

st.markdown("---")
st.markdown("© 2026 RAG Document Buddy — built locally, powered by Llama 3.2 🦙")