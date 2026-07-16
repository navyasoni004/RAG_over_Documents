# vectors.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma


class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        persist_dir: str = "chroma_db",
        collection_name: str = "vector_db",
    ):
        """
        Handles PDF loading, chunking, embedding, and storage in a local
        Chroma vector store (no Docker/server required).
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def create_embeddings(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # Load and preprocess the PDF (lightweight, no OCR deps)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No documents were loaded from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)

        if not splits:
            raise ValueError("No text chunks were created from the documents.")

        # Create and persist embeddings locally with Chroma
        try:
            Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create vector store: {e}")

        return "✅ Vector DB Successfully Created and Stored Locally (Chroma)!"