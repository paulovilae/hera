# 🛡️ Official Capabilities Matrix: VETRA vs HERA vs MEMENTO

> **🚨 MANDATORY READING FOR ALL AI AGENTS (Vetra, Latinos, Salarium) 🚨**
>
> **Context:** It was detected that agents working in Vetra attempted to create custom PDF generators using Node dependencies. This is a direct violation of the Vilaros Sovereign Architecture.
>
> **The Tripartite Rule:** The architecture is divided into three distinct brains: **Vilaros/Vetra** (Transactional UI/Postgres), **Hera Core** (Active Inference & Tools in Rust), and **Memento** (Long-Term Vector Memory & RAG in Rust).
>
> **Golden Rule:** **Vetra NEVER performs heavy computational tasks, document manipulation, or AI Inference.** Vetra is exclusively a Frontend Interface and Relational Database Management layer (Drizzle ORM). All heavy lifting tools already live compiled in **Hera Core (Rust)**.

Below is the absolute matrix of responsibilities. **Agents are strictly forbidden from installing packages, creating modules, or reinventing capabilities that belong in the Hera Core or Memento columns.**

---

## 🛑 1. Document Generation & Manipulation (PDFs, Excel)

| Capability | Who does it? | How is it invoked? | Forbidden for the other to do: |
| :--- | :--- | :--- | :--- |
| **PDF Generation** from structured data | **HERA CORE** | `POST http://localhost:3305/v1/hera/generate-pdf` sending the contract JSON. | Vetra is forbidden from installing libraries like `pdfkit`, `jspdf`, `puppeteer`, or trying to draw PDFs locally. |
| **Excel/CSV Extraction** | **HERA CORE** | `POST /v1/hera/extract` (Upload the Excel, Hera parses it via native Calamine and returns JSON). | Vetra is forbidden from installing `xlsx`, `exceljs` or parsing binaries. |
| **OCR of Images/Scanned PDFs** | **HERA CORE** | `POST /v1/hera/extract` (Hera applies OCR and passes it to Moondream/Qwen to structure). | Vetra is forbidden from using Tesseract or sending images to third-party APIs. |

---

## 🎨 2. Multimedia Generation (Images, Video, Audio)

| Capability | Who does it? | How is it invoked? | Forbidden for the other to do: |
| :--- | :--- | :--- | :--- |
| **Image Generation** (SwarmUI/Flux/ComfyUI) | **HERMES NODE** | Call via the OpenClaw / Hermes RPC network or standard T2I endpoints. | Vetra must NEVER render images client-side or use native Next.js server-side canvas wrappers. |
| **Video Generation** (LTX-2) | **HERMES NODE** | Sent via Hermes API (heavy RTX 3090 task). | Vetra must strictly serve as a video *player* (UI), never a generator or transcoder. |
| **TTS / Speech Generation** (Piper) | **OPENCLAW / HERMES** | Called via HTTP TTS endpoints or natively synthesized by OpenClaw. | Vetra must NOT attempt to run local browser TTS APIs for authoritative audio, or wrap `ffmpeg` in Node. |
| **Song / Music Generation** | **HERA / EXTERNAL NODE** | Handled natively by Python/Rust compute nodes. | Vetra must NOT install audio synthesizers in `vetra3`. |

---

## 🧠 3. Artificial Intelligence and Natural Language Processing

| Capability | Who does it? | How is it invoked? | Forbidden for the other to do: |
| :--- | :--- | :--- | :--- |
| **AI Chat, Analysis, and Generation** (OpenAI Compat) | **HERA CORE** | Calls to `http://localhost:3305/v1/chat/completions` (OpenAI SDK). | Vetra must never call OpenAI or Anthropic directly. ALL requests pass through Hera for sovereign "Failover". |
| **Logic Tools / Output Formatting** | **HERA CORE** | Send prompt with `response_format` JSON Schema to Hera. | Vetra must not try to use massive regex on text to guess fields if Hera can do it. |
| **Web Search & Scrape** | **HERA CORE** | `POST /v1/hera/search` & `POST /v1/hera/scrape`. | Vetra must never use `axios`/`cheerio` to scrape external pages. Route all scrape intents through Hera. |
| **Recursive Language Models (RLM)** | **HERA CORE** | Hera sets up REPL environments to recursively call sub-LMs to parse Memento. | Vetra must never attempt to build programmatic REPL environments for LLM inference loops. |

---

## 🗄️ 4. Episodic Memory and Embeddings (RAG)

| Capability | Who does it? | How is it invoked? | Forbidden for the other to do: |
| :--- | :--- | :--- | :--- |
| **Vector Embedding Generation** | **MEMENTO** | Accessed by Hera via Zero-Copy UDS. | Hera must not run heavy background embeddings while performing chat inference. Vetra must absolutely not embed text. |
| **RAG / Semantic Similarity** | **MEMENTO** | Queried by Hera against Qdrant natively via Memento. | Vetra must not install Langchain or LlamaIndex. All semantic memory lives in Memento. |
| **Conversation Archiving** | **MEMENTO** | Hera sends asynchronous payloads to Memento to store. | Vetra only saves *business entities* (Contracts, Users) in Postgres, NEVER raw AI conversation logs to be queried semantically. |

---

## 🏗️ 5. Exclusive Responsibilities of VETRA (The Client)

If you are an agent coding in the `vetra3`, `latinos`, or `vilaros` folders, these are your ONLY areas of work:

1. **Frontend / UI (Global Corporate):** React, Next.js, Tailwind CSS (Strictly Tailwind, no ad-hoc CSS), Zustand/Context for state. Vilaros is the absolute owner of ALL *Central/Corporate* user interfaces, including global memory dashboards or RLM trajectories.
   * *Exception:* **Local Memento Nodes** (running on individual user laptops) are permitted to serve a **Local Device Admin UI** (e.g., via `localhost` Tauri or Axum) strictly for managing local folder ingestion, GPU health, and PII sanitization. They must NOT replicate Vilaros business logic.
2. **Database / Drizzle ORM:** Vetra is the absolute OWNER of the relational schema definition (Postgres). If a data model changes, it changes in Vetra.
3. **Authentication (Auth):** Integration with NextAuth and Authentik OIDC.
4. **Next.js Controllers (API Routes):** They serve *exclusively* as a Bridge to orchestrate calls to Hera Core and save the results in Postgres. They must not contain heavy logic.
5. **Schema Definition (Zod):** Vetra defines *what* data it needs (e.g., The contract schema) and sends it to Hera so Hera works on it.

---

## 👨‍💻 6. Summary for Agents: "The Tripartite Brain"

*   **VETRA = UI & Business State (Postgres/Drizzle)**. Vetra knows *what* information to show to the user and *where* relations are saved.
*   **HERA = The Muscle and the Inference (Rust)**. Hera knows *how* to solve immediate problems, parse files, run tools, and think (LLM).
*   **MEMENTO = The Long-Term Memory (Rust/Qdrant)**. Memento knows *who* the user is across time, what was said historically, and manages semantic vectors.

> **Example:** If you are in Vetra and are asked to "Generate a PDF contract", **DO NOT CREATE A PDF ENGINE**. Your job is to fetch the data from Postgres with Drizzle, assemble the JSON, send it to the `/v1/hera/generate-pdf` endpoint in Hera, and return the resulting file to the Frontend.
