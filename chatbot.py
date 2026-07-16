# chatbot.py
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
import streamlit as st

# Load HUGGINGFACEHUB_API_TOKEN from .env
load_dotenv()


class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_repo_id: str = "Qwen/Qwen2.5-7B-Instruct",
        llm_temperature: float = 0.7,
        persist_dir: str = "chroma_db",
        collection_name: str = "vector_db",
    ):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs=encode_kwargs,
        )

        # Remote LLM via HuggingFace Inference API — nothing downloaded locally
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN not found. Add it to a .env file "
                "in the project root."
            )

        endpoint = HuggingFaceEndpoint(
            repo_id=llm_repo_id,
            huggingfacehub_api_token=hf_token,
            temperature=llm_temperature,
            max_new_tokens=512,
        )
        # ChatHuggingFace wraps the endpoint so it behaves like a chat model
        self.llm = ChatHuggingFace(llm=endpoint)

        # Prompt template — forces grounded, no-hallucination answers
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"],
        )

        # Reconnect to the same local Chroma store built by vectors.py
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        # k=3 retrieves top 3 chunks instead of just 1 — better context
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt},
            verbose=False,
        )

    def get_response(self, query: str) -> str:
        try:
            response = self.qa.invoke({"query": query})
            return response["result"]
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."