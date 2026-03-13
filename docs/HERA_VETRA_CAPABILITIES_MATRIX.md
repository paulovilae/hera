# 🛡️ Official Capabilities Matrix: VETRA vs HERA

> **🚨 MANDATORY READING FOR ALL AI AGENTS (Vetra, Latinos, Salarium) 🚨**
>
> **Context:** It was detected that agents working in Vetra attempted to create custom PDF generators using Node dependencies. This is a direct violation of the Vilaros Sovereign Architecture.
>
> **Golden Rule:** **Vetra NEVER performs heavy computational tasks, document manipulation, or AI Inference.** Vetra is exclusively a Frontend Interface and Relational Database Management layer (Drizzle ORM). All heavy lifting tools already live compiled in **Hera Core (Rust)**.

Below is the absolute matrix of responsibilities. **Agents are strictly forbidden from installing packages, creating modules, or reinventing capabilities that belong in the Hera Core columns.**

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
| **Complex Mappings / RAG** | **HERA CORE** | Structured interface of Tools/Functions against Hera Core. | Vetra must not try to program algorithmic RAG or NLP logic; that belongs to Hera. |
| **Structured Text Extraction (JSON)** | **HERA CORE** | Send prompt with `response_format` JSON Schema to Hera. | Vetra must not try to use massive regex on text to guess fields if Hera can do it. |
| **Web Search & Scrape** | **HERA CORE** | `POST /v1/hera/search` & `POST /v1/hera/scrape`. | Vetra must never use `axios`/`cheerio` to scrape external pages. Route all scrape intents through Hera. |

---

## 🏗️ 4. Exclusive Responsibilities of VETRA (The Client)

If you are an agent coding in the `vetra3` or `latinos` folder, these are your ONLY areas of work:

1. **Frontend / UI:** React, Next.js, Tailwind CSS (Strictly Tailwind, no ad-hoc CSS), Zustand/Context for state.
2. **Database / Drizzle ORM:** Vetra is the absolute OWNER of the relational schema definition (Postgres). If a data model changes, it changes in Vetra.
3. **Authentication (Auth):** Integration with NextAuth and Authentik OIDC.
4. **Next.js Controllers (API Routes):** They serve *exclusively* as a Bridge to orchestrate calls to Hera Core and save the results in Postgres. They must not contain heavy logic.
5. **Schema Definition (Zod):** Vetra defines *what* data it needs (e.g., The contract schema) and sends it to Hera so Hera works on it.

---

## 👨‍💻 5. Summary for Agents: "Think vs Do"

*   **VETRA = UI & Long-term Memory (Postgres/Drizzle)**. Vetra knows *what* information to show and *where* to save it.
*   **HERA = The Muscle and the Brain (Rust)**. Hera knows *how* to process files, *how* to generate PDFs, and *how* to think (LLM).

If you are in Vetra and are asked to "Generate a PDF contract", **DO NOT CREATE A PDF ENGINE**. Your job is to fetch the data from Postgres with Drizzle, assemble the JSON, send it to the `/v1/hera/generate-pdf` endpoint in Hera, and return the resulting file to the Frontend.
