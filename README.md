graph TB
    subgraph UI["🖥️ Presentation Layer · Streamlit"]
        UPLOAD["📄 PDF Upload Widget"]
        CHAT["💬 Chat Interface"]
        EMB_BTN["☐ Create Embeddings Toggle"]
    end

    subgraph INGEST["📥 Ingestion Pipeline"]
        LOADER["PyPDFLoader"]
        SPLITTER["RecursiveCharacterTextSplitter<br/>chunk_size=1000"]
        EMBED["BAAI/bge-small-en<br/>CPU Embeddings"]
        STORE["Chroma Collection"]
        LOADER --> SPLITTER --> EMBED --> STORE
    end

    subgraph RETRIEVE["🔍 Retrieval Pipeline"]
        Q_EMBED["Query Embedder"]
        SIM["Similarity Search<br/>top-k = 3"]
        RANK["Context Assembler"]
        Q_EMBED --> SIM --> RANK
    end

    subgraph GEN["🧠 Generation Layer"]
        PROMPT["Prompt Template"]
        LLM["Qwen2.5-7B-Instruct"]
        CHAIN["RetrievalQA Chain"]
        PROMPT --> CHAIN
        LLM --> CHAIN
    end

    subgraph STORE_LAYER["💾 Storage & Config"]
        CHROMA_PERSIST[("🗄️ ChromaDB")]
        ENV["🔐 .env"]
    end

    subgraph EXT["☁️ External Service"]
        HF_API["HuggingFace Cloud"]
    end

    UPLOAD -->|"file bytes"| LOADER
    EMB_BTN -->|"trigger"| EMBED
    STORE <-->|"read/write"| CHROMA_PERSIST
    EMBED -.->|"uses key"| ENV

    CHAT -->|"user question"| Q_EMBED
    STORE -->|"similarity query"| SIM
    RANK -->|"context + question"| PROMPT
    CHAIN -->|"POST / inference"| HF_API
    HF_API -->|"generated text"| CHAIN
    CHAIN -->|"grounded answer"| CHAT