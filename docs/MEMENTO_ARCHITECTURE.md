# 🧠 Memento: Sovereign Memory Architecture
**Status:** Architecture Draft

**Memento** is the dedicated Memory Agent ("Hippocampus") for the ImagineOS ecosystem. It is designed as a **Distributed P2P Sovereign Mesh** handling all Vector/Semantic/Episodic Memory, background summarization, and RAG embeddings.

Crucially, Memento enables the **Recursive Language Model (RLM)** paradigm. It acts as the infinite-context backend, moving away from simple linear vector-stuffing and allowing the inference engine (Hera) to programmatically query, decompose, and recursively synthesize historical knowledge.

---

## 🏗️ The Tripartite Architecture
1. 🖥️ **Vilaros (Prefrontal Cortex):** UI, Auth, Relational State (Postgres via Drizzle). Knows *what* to show to the user.
2. 🦾 **Hera (Motor/Analytical Cortex):** Real-time Inference (LLM), OCR, Document Parsing, Tools. Knows *how* to solve tasks immediately.
3. 🧠 **Memento (Hippocampus):** Long-term Episodic Memory, RAG Context, Embeddings. Remembers *who* the user is, *what* was said, and *where* facts are stored.

---

## ⚡ High-Performance IPC (Inter-Process Communication)
Because Memento and Hera run on the same sovereign local cluster, they **do not** use standard HTTP/REST or standard TCP loops, which present thousands of microseconds in overhead due to serialization, network stacks, and context switches.

Instead, Memento and Hera use **Zero-Copy IPC**:

### 1. Transport Layer: Unix Domain Sockets (UDS)
Connections between Memento and Hera are routed through local `.sock` files on the host filesystem (`/tmp/memento.sock`). Unix sockets bypass the OS network stack entirely, bringing latency as close to bare metal as possible.

### 2. Data Layout: Apache Arrow / rkyv (Zero-Copy)
To avoid standard JSON translation (Serialize String -> Deserialize String -> Allocate memory -> Read), Memento writes data in a guaranteed binary schema using **rkyv** or **Apache Arrow Flight**:
* **rkyv:** Memento writes the memory payload to the socket. Hera reads the raw bytes and *instantly* casts a Rust reference (`&ArchivedMemoryPayload`) directly over those bytes. Zero allocations, zero copies.
* **Result:** Hera can query a 10MB chunk of past user conversations from Memento and read it in nanoseconds, passing it directly into the LLM context window.

---

## 💾 Storage Layer (The Hybrid Memory System)
Memento does not rely solely on one type of memory. It uses a **Hybrid Memory System** to ensure it never forgets core facts (like "The user prefers Rust") while still being able to search thousands of documents.

### 1. Semantic Knowledge Graph (The Core Facts)
* **What it is:** A structured JSON/Graph database (often stored natively or in Postgres/Neo4j). 
* **Purpose:** It stores atomic instructions, preferences, and undeniable facts about the user (e.g., `user.language_preference: "Rust"`, `user.name: "Paulo"`).
* **Behavior:** Before Hera answers *any* prompt, Memento performs a fast lookup on this graph. If a core fact is triggered, it is **injected directly into the system prompt**. This guarantees Hera *never* forgets basic stuff, because the facts bypass vector probability and become absolute rules.

### 2. Episodic Vector Memory (Qdrant)
* **What it is:** The massive Qdrant vector database managing local embeddings (e.g., FastEmbed).
* **Purpose:** Archiving long conversations, PDFs, codebases, and historical workflows.
* **Behavior:** This is used for "fuzzy" recall. When Hera needs to remember "How did we solve that Docker port issue last month?", Memento runs a semantic similarity search across Qdrant to pull the exact conversational episode.

### 3. The Write Path (Synthesis)
When Hera finishes a thought, it fires an asynchronous payload to Memento. Memento runs two parallel jobs:
1. **Extraction Pipeline:** It asks a tiny sub-agent: *"Did the user state a hard fact or preference here?"* If yes, update the **Semantic Graph**.
2. **Embedding Pipeline:** Compute the vectors for the raw conversation and push it to **Qdrant**.

---

## 🪆 Recursive Language Models (RLM) & Infinite Context
ImagineOS embraces the RLM paradigm (replacing standard linear context windows). Because context size limits cognitive reasoning logic, Hera must not receive raw "stuffed" text.

1. **Hera (The Executor):** Hera maintains REPL environments (like DockerREPL or LocalREPL) to run sub-LMs.
2. **Memento (The Variable):** The user's entire history and world-knowledge is offloaded to Memento. Memento acts as a searchable REPL variable.
3. **The Loop:** When Hera receives a complex query requiring years of context, it writes a small program (or uses tool calls) to recursively launch sub-LMs. These sub-LMs query Memento, summarize the relevant chunks, and return distilled insights to the root Hera process.

---

## 🌐 The Distributed Mesh (P2P Sovereign Memory)
Memento is not a single database attached to Hera; it is a **Distributed Protocol**. A company or user can run multiple Memento nodes across different devices:

1. **Central Memento (The Server):** Runs alongside Hera in the cloud, storing global corporate knowledge and historical chat.
2. **Local Memento (The Laptop):** Runs on a user's workstation. It watches local folders (e.g., `/home/paulo/Documents`).
3. **Local Conversion Engine:** Hera *never* receives raw documents. All parsing of PDFs, Word (DOCX), and Excel (XLSX) files happens 100% locally on the Memento node using localized memory allocation. The local engine extracts the text offline before passing it to the Privacy Shield.
4. **Cloud Storage Connectors:** In addition to local folders, Local Memento runs native API pullers (using OAuth via Authentik) to ingest documents from **Google Drive, OneDrive, and Nextcloud**. It pulls these cloud documents down to the local machine, vectorizes them locally, and adds them to the privacy queue.
5. **The Connection (WireGuard/gRPC Handshake):** When Hera needs to answer a prompt, it sends a high-speed gRPC request to the user's Local Memento over the **GCP Caddy + WireGuard Bridge**. The local node authenticates Hera's request using OIDC tokens, executes the vector search locally, and returns *only* the sanitized text chunks. Your personal files never leave the laptop until explicitly queried by an authenticated Hera node.

### 🛡️ The Privacy Shield (Local Sanitization)
To support strict enterprise compliance, **Local Mementos run a localized SLM (Small Language Model)** (e.g., Llama-3-8B or Qwen-1.5B) directly on the user's device.
* **Optional & Asynchronous:** Because SLM inference is computationally expensive, sanitization is not strictly real-time and can be toggled per-folder. Local Memento can operate as a background agent—waking up overnight or during idle hours to bulk-sanitize selected document batches without interrupting the user's daily workflow.
* **The Process:** During the sanitization pass, the Local SLM processes each chunk: *"Strip all PII, passwords, and API keys"* before the semantic vector is finalized or permitted to be queried by the central Hera node.
* **The Result:** The Central Hera AI only receives sanitized, semantic representations of local documents, ensuring zero data leakage.

### ⚙️ Hardware Auto-Scaling (Liquid Compute)
Because Local Mementos run on heterogeneous hardware, the Rust binary performs auto-discovery on startup:
1. **GPU Fast Path:** If Memento detects an NVIDIA RTX (CUDA) or Apple Silicon (Metal), it loads the SLM and FastEmbed models into VRAM for blazing-fast local sanitization and embedding.
2. **CPU Fallback:** If no GPU is found, Memento gracefully falls back to quantized ONNX/CPU execution using minimal RAM, ensuring it runs invisibly in the background.

---

## 🎨 Frontend & UI Ownership
The Sovereign Architecture enforces strict UI rules regarding memory management:

1. **Vilaros (The Central Hub):** Vilaros remains the absolute owner of the **Global UI**. Users log into Vilaros to view massive Memory Graphs, RLM trajectories, and tune corporate RAG weights.
2. **Local Device UI (The Memento Dashboard):** Because Local Mementos run on individual laptops, they expose a lightweight, standalone UI (e.g., `localhost:3306` or a Tauri app).
   * **Purpose:** This local UI allows the user to select which folders to index, review what PII was stripped by the Local SLM, and check local GPU/CPU health. It *does not* replicate the Vilaros business logic.
