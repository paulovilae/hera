# 🛡️ Hera-Vetra Integration Guide (For the Vetra Agent)

Hello Vetra developer (or Vetra Agent),

Hera Core has been significantly updated as part of our migration towards the **Vilaros Sovereign Architecture**. If you previously interacted with Hera through the Node/Python stack or via legacy endpoints, here is everything you need to know about the new capabilities exposed by Hera, which now run natively on an ultra-fast Rust engine (Loco.rs/Axum) ported on `3305`.

## 📌 Restored Endpoint: `/v1/hera/extract` (Universal Extraction)
**Route:** `POST /v1/hera/extract`
*   **Purpose:** This is the main endpoint ("Universal Extraction") for processing PDFs, Excels, or Images, extracting data into a predefined JSON schema using the sovereign multimodal LLM engine located in Hera (Moondream + Qwen). 
*   **Updated Behavior:** This endpoint was temporarily unavailable due to severe refactoring to support the universal `Anthropic/OpenAI` format, but it has been **restored** in this version. Vetra3 must connect here for any KYB flow (Chamber of Commerce / RUT uploads).
*   **Payload:** It still expects the upload in `multipart/form-data` format containing the respective file and the request parameters.

## 📌 New Endpoint: `/v1/hera/generate-pdf` (Universal PDF Generation)
**Route:** `POST /v1/hera/generate-pdf`
*   **Purpose:** Takes a database schema (for example, data coming from contracts in Vetra ready for review) and automatically renders them into a native PDF buffer.
*   **Return:** Depending on the `"return_base64"` flag, it returns either raw bytes (`application/pdf`) ready for download/visualization, or a `Base64` string wrapping the payload. This is done entirely within Hera's internal memory.

## 📌 Universal Compatibility (`/v1/chat/completions` and `/v1/messages`)
*   Hera-Core is now a 100% compatible proxy with clients expecting to talk to the **OpenAI API** or the **Anthropic API**.
*   **OpenAI:** Use `POST /v1/chat/completions`.
*   **Anthropic:** Use `POST /v1/messages`.
*   This means you NO LONGER need to build complex wrappers in the Vetra3 backend; simply use standard libraries like the OpenAI SDK client in TypeScript pointing to `http://localhost:3305/v1` (or in production according to `.env.local`).

## ⚙️ Critical Debugging Tips:
1.  **LLM VRAM:** Hera-Core can take **> 35-50 seconds** for the first cold start when extracting documents (`/v1/hera/extract`), due to the loading of the Local model into the GPU (Dual RTX 3090). **It is not an error (Fetch Failed)** if you experience a timeout, make sure the Vetra3 backend is not killing the connection prematurely. Increase your fetch timeouts to 120+ seconds.
2.  **Logs:** The logs to investigate what Hera is doing are reviewed directly using the command `pm2 logs hera-core`.

Sincerely,
**Antigravity (Core Engineer)**
