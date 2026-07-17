# 📚 RAG over Documents

A local-first **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload a PDF and ask questions about it. Combines local embeddings, a local vector store, and a cloud-hosted LLM for generation.

Built as a hands-on exploration of the RAG pipeline — document ingestion, chunking, embedding, vector search, and grounded LLM generation — end to end.

---

## 🧠 Architecture

```mermaid
flowchart TB
    subgraph CLIENT["🖥️ CLIENT LAYER"]
        direction TB
        U1(["👤 User · Browser"]) --> U2["Streamlit Session State<br/>chat_history · uploaded_file"]
    end

    subgraph INGEST["📥 INGESTION PIPELINE — runs once, on upload"]
        direction TB
        I1["📄 Uploaded PDF<br/>raw bytes, temp path"] --> I2["PyPDFLoader.load()<br/>→ List[Document]"]
        I2 --> I3["Per-page metadata<br/>source, page_number"]
        I3 --> I4["RecursiveCharacterTextSplitter<br/>chunk_size=1000, overlap=250"]
        I4 --> I5["List[Document] chunks<br/>+ metadata"]
    end

    subgraph EMBED["🧬 EMBEDDING LAYER — CPU, local"]
        direction TB
        E1["HuggingFaceEmbeddings<br/>model = BAAI/bge-small-en"] --> E2["Tokenize + forward pass<br/>sentence-transformers"]
        E2 --> E3["384-dim dense vectors"]
    end

    subgraph STORE["🗄️ VECTOR STORAGE LAYER — local disk"]
        direction TB
        S1[("Chroma.from_documents()<br/>persist_directory = ./db")] --> S2["SQLite + Parquet index<br/>on-disk, persistent"]
    end

    subgraph RETRIEVE["🔍 RETRIEVAL PIPELINE — runs once, per message"]
        direction TB
        R1(["❓ User Question"]) --> R2["HuggingFaceEmbeddings<br/>same model instance"]
        R2 --> R3["Query vector · 384-dim"]
        R3 --> R4{{"vectorstore.similarity_search()<br/>k = 3, cosine distance"}}
        R4 --> R5["Top-3 ranked chunks<br/>+ similarity scores"]
    end

    subgraph PROMPT["📝 PROMPT CONSTRUCTION"]
        direction TB
        P1["Context assembly<br/>join(chunks, sep = '\n\n')"] --> P2["PromptTemplate.format()<br/>system + context + question"]
        P2 --> P3["Grounding instruction:<br/>'answer only from context,<br/>else say I don't know'"]
    end

    subgraph GEN["☁️ GENERATION LAYER — the one network hop"]
        direction TB
        G1["RetrievalQA chain<br/>.invoke()"] --> G2["HTTPS request →<br/>api-inference.huggingface.co"]
        G2 --> G3{{"Qwen2.5-7B-Instruct<br/>hosted inference endpoint"}}
        G3 --> G4["Response parsing +<br/>rate-limit / retry handling"]
        G4 --> G5["Grounded answer string"]
    end

    subgraph OUTPUT["💬 RESPONSE LAYER"]
        direction TB
        O1["chat_history.append()"] --> O2(["Rendered in Streamlit UI"])
    end

    subgraph CONFIG["⚙️ CONFIG & SECRETS"]
        direction TB
        CFG1[".env<br/>HUGGINGFACEHUB_API_TOKEN"]
        CFG2["requirements.txt<br/>langchain · chromadb<br/>sentence-transformers · streamlit"]
    end

    U2 -->|"upload + 'Create Embeddings'"| I1
    U2 -->|"type question"| R1
    I5 --> E1
    E3 -->|upsert vectors| S1
    S2 -. indexed vectors .-> R4
    R5 --> P1
    P3 --> G1
    G5 --> O1
    O2 -. next turn .-> R1
    CFG1 -. loads via python-dotenv .-> G2
    CFG2 -. pins deps .-> E1
    CFG2 -. pins deps .-> G1

    classDef client fill:#111a27,stroke:#7c8ca0,color:#e8eef5,stroke-width:2px;
    classDef idx fill:#1b2a3f,stroke:#5b8def,color:#e8eef5,stroke-width:2px;
    classDef emb fill:#142a28,stroke:#3fb8a8,color:#e8eef5,stroke-width:2px;
    classDef hub fill:#0e1e1c,stroke:#43d9c4,color:#e8eef5,stroke-width:2px;
    classDef qry fill:#2a2415,stroke:#e8a33d,color:#e8eef5,stroke-width:2px;
    classDef prompt fill:#241f38,stroke:#b39ddb,color:#e8eef5,stroke-width:2px;
    classDef cloud fill:#1c1730,stroke:#9a87e0,color:#e8eef5,stroke-width:2px;
    classDef out fill:#111a27,stroke:#7c8ca0,color:#e8eef5,stroke-width:2px;
    classDef cfg fill:#1a1410,stroke:#c98a3d,color:#e8eef5,stroke-width:1.5px,stroke-dasharray: 3 3;

    class U1,U2 client;
    class I1,I2,I3,I4,I5 idx;
    class E1,E2,E3 emb;
    class S1,S2 hub;
    class R1,R2,R3,R4,R5 qry;
    class P1,P2,P3 prompt;
    class G1,G2,G3,G4,G5 cloud;
    class O1,O2 out;
    class CFG1,CFG2 cfg;
```

Nine layers, two triggers:

| Layer | Fires on | What it owns |
|---|---|---|
| 🖥️ Client | every interaction | Streamlit session state — chat history, uploaded file |
| 📥 Ingestion | PDF upload + "Create Embeddings" | `PyPDFLoader` → `RecursiveCharacterTextSplitter` (1000/250) |
| 🧬 Embedding | ingestion **and** every query | `bge-small-en`, local, CPU-only, same instance both times |
| 🗄️ Storage | written once, read every query | Chroma, persisted to disk as SQLite + Parquet |
| 🔍 Retrieval | every message | cosine similarity search, top-k = 3 |
| 📝 Prompt construction | every message | assembles context + grounding instruction |
| ☁️ Generation | every message | the **only** network call — Qwen2.5-7B-Instruct via HuggingFace Inference API |
| 💬 Response | every message | writes to chat history, renders, loops back for the next turn |
| ⚙️ Config | startup | `.env` token + pinned `requirements.txt`, feeding both the embedding and generation layers |

No Ollama, no locally hosted LLM weights — the embedding model is the only thing that runs on-machine; generation is a single HTTPS round trip. The system is explicitly prompted to answer **only from retrieved context** and say "I don't know" otherwise — reducing hallucination on out-of-scope questions.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| PDF Parsing | pypdf |
| Embeddings | HuggingFace `BAAI/bge-small-en` (local, CPU) |
| Vector Store | Chroma (local, persistent) |
| LLM | Qwen2.5-7B-Instruct via HuggingFace Inference API |
| Secrets | python-dotenv |

**Design choice:** embeddings run locally (small, fast, free) while generation calls a hosted API (avoids downloading multi-GB model weights, works on CPU-only machines).

---

## 📂 Project Structure

```
rag-buddy/
├── app.py              # Streamlit UI — wires everything together
├── vectors.py           # PDF loading, chunking, embedding → Chroma
├── chatbot.py            # Retrieval + prompt + LLM call
├── requirements.txt
├── .env.example          # Template for required API key
└── .gitignore
```

| File | Responsibility |
|---|---|
| `vectors.py` | `EmbeddingsManager` — loads a PDF, splits it into overlapping chunks, embeds them, and persists them to a local Chroma collection. |
| `chatbot.py` | `ChatbotManager` — reconnects to the same Chroma collection as a retriever, defines the grounded prompt template, and calls the HF-hosted LLM through a `RetrievalQA` chain. |
| `app.py` | Streamlit app — handles file upload, triggers embedding creation, renders chat history, and calls `ChatbotManager.get_response()` per message. |

---

## 🚀 Setup

### 1. Clone and create environment
```bash
git clone https://github.com/navyasoni004/rag-document-buddy.git
cd rag-document-buddy
conda create --name ragbuddy python=3.10
conda activate ragbuddy
pip install -r requirements.txt
```

### 2. Get a HuggingFace API token
Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough).

### 3. Configure environment variables
```bash
cp .env.example .env
```
Then edit `.env` and add your token:
```
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

### 4. Run
```bash
streamlit run app.py
```
Open `http://localhost:8501` → **Chatbot** page → upload a PDF → check "Create Embeddings" → ask questions.

---

## ⚠️ Notes & Limitations

- Works best on text-based PDFs; scanned/image-only PDFs need OCR (not included — see `pytesseract` for an extension).
- Free-tier HuggingFace Inference API has rate limits and can be slow during peak load.
- `k=3` retrieval is a tunable trade-off — more chunks give the LLM more context but increase prompt size and latency.

---

## 🔮 Possible Extensions

- Add a reranker (e.g. cross-encoder) after retrieval for better relevance
- Support multi-document / multi-PDF sessions
- Swap Chroma for a hosted vector DB for multi-user deployment
- Add source citations (which chunk/page an answer came from)
