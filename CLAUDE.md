# CLAUDE.md — Hera

This file provides guidance when working inside the `Hera/` submodule.

---

## Submodule Rules

Hera is a git submodule tracked by the parent OS repo. The rules are:

1. **Always commit inside Hera first.** Make changes, `git add`, `git commit` inside `Hera/`. Only then go to the OS root and update the parent pointer (`git add Hera && git commit -m "chore: update Hera pointer"`).
2. **Never modify Hera files and commit only from the OS root.** The parent only tracks a commit hash. If you commit from the OS root without committing inside Hera first, the submodule will show as dirty forever.
3. **`+` prefix in `git submodule status`** means the checked-out commit differs from what the parent records. Resolve by committing inside Hera and updating the parent pointer, not by discarding Hera changes.

Publication target: `paulovilae/hera`. Edition: **2024** (Hera's Cargo edition is `2024`; if-let chains and let-else work here — OS-v3 is edition 2021 and does not.)

---

## Architecture (as of 2026-05-27)

```
Hera/
├── hera-core/src/
│   ├── ai/
│   │   ├── mod.rs                       — LLMEngine trait, ChatRequest/ChatResponse types
│   │   │                                    (response_format pass-through for structured outputs — frame F)
│   │   ├── router.rs                    — RouterEngine: primary → secondary → tertiary → cloud fallback
│   │   ├── tool_executor/               — split into mod/dispatch/registry/schema/security/intent
│   │   ├── tools/                       — per-app and global tool modules (data, productivity, infra_*, apps_*)
│   │   ├── embeddings.rs                — candle BERT embeddings (CPU, mean-pool+L2)            [feature `embeddings`]
│   │   ├── engine_gguf.rs               — local GGUF/llama.cpp engine                            [feature `local-llm`]
│   │   ├── native_engine.rs             — native inference engine                                [feature `local-llm`]
│   │   ├── gemini.rs                    — Gemini cloud engine
│   │   └── openai_compat.rs             — OpenAI-compatible frontend
│   ├── ipc/
│   │   ├── mod.rs                       — dispatcher (action → handler_*)
│   │   ├── context.rs                   — ParsedPayload, build_full_system_prompt (cache-friendly, frame E),
│   │   │                                    prepare_runtime_execution_context (calls difficulty classifier)
│   │   ├── difficulty.rs                — hybrid difficulty classifier (heuristic + embedding tiebreak), frame B
│   │   ├── helpers.rs                   — Memento IPC client (call_memento), canonicalize_user_id,
│   │   │                                    fetch_recursive_context, fetch_semantic_memories,
│   │   │                                    save_chat_turn_event, embed_text_local (feature-gated)
│   │   ├── handler_generate.rs          — non-streaming generate + B3 quality cascade
│   │   ├── handler_stream.rs            — streaming generate + B3 (post-stream)
│   │   ├── handler_embed.rs             — IPC `embed` action (uses ai::embeddings)               [feature `embeddings`]
│   │   ├── route_profiles.rs            — per-route SLO + budget mode mapping
│   │   ├── runtime_tools.rs             — tool execution helpers
│   │   └── handler_* (delegation/dag/media/audio/tools/health/lora)
│   ├── bin/
│   │   ├── claude.rs                    — local prompt CLI (routes through hera-core.sock)
│   │   ├── hera_mcp.rs                  — Hera as an MCP server (rmcp)                           [feature `mcp-bridge`]
│   │   └── ...
│   ├── main.rs                          — daemon startup, socket listener
│   └── rest_api.rs                      — REST API surface (port 3002, fallback only)
└── scripts/
    ├── start_native_omni.sh             — starts the local GGUF model server
    └── start_native_draw.sh             — starts the image generation backend
```

---

## Cargo Features

Three relevant features are layered to support heterogeneous nodes:

| Feature        | Pulls in                                                   | Use on                                  |
|----------------|------------------------------------------------------------|-----------------------------------------|
| `embeddings`   | `candle-core`, `candle-transformers`, `candle-nn`, `tokenizers`, `hf-hub` (CPU; no CUDA) | Any node that should run semantic recall locally (currently anchor, CPU-only) |
| `cuda`         | `candle-*/cuda`, `llama-cpp-2/cuda`                        | GPU sub-feature; activated by `local-llm` |
| `local-llm`    | `embeddings` + `cuda` + `hound` + `llama-cpp-2`            | GPU nodes (genesis, atlas) that run a local LLM |
| `mcp-bridge`   | `rmcp`, `schemars`                                         | When running `hera_mcp` as an MCP server |

**Important**: candle's `cuda` is **not** hardcoded on the dependency anymore — it's a sub-feature. So `embeddings` builds candle on CPU without requiring CUDA, which is what lets CPU-only nodes have semantic recall.

---

## IPC Socket

Hera listens on `/tmp/hera-core.sock` (Unix Domain Socket). All app communication goes here.

**Live actions:**

| Action             | Purpose                                                                  | Where                          |
|--------------------|--------------------------------------------------------------------------|--------------------------------|
| `generate`         | Non-streaming chat (with tool execution)                                 | `handler_generate`             |
| `generate_stream`  | Streaming chat                                                           | `handler_stream`               |
| `execute_tool`     | Direct tool execution                                                    | `handler_tools::handle_execute_tool` |
| `embed`            | Turn text(s) into vectors (sentence-transformers MiniLM-L12 multilingual)| `handler_embed` (feature `embeddings`) |
| `delegate_task`    | Spawn sub-agent runs                                                     | `handler_delegation`           |
| `route_health`     | Per-route latency/error report                                           | `handler_health`               |
| `generate_image`, `vision_analysis`, `transcribe_audio`, `execute_dag`, `get_tools`, ... | various | `handler_media`, etc. |

**Request format (generate):**
```json
{
  "action": "generate",
  "payload": {
    "prompt": "...",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 800,
    "temperature": 0.3,
    "permissions": ["garcero"],
    "session_id": "...",
    "chat_id": "...",
    "sender_name": "...",
    "app": "garcero",
    "route_profile": "garcero_widget",
    "response_format": {"type": "json_schema", "json_schema": {...}}
  }
}
```

`response_format` (frame F) is pass-through to the engine; both llama.cpp local and OpenAI-compatible cloud honor it. Use it when you need a strict JSON shape (e.g. the Studio artifact dispatch).

---

## Router Engine (`ai/router.rs`)

Four-tier fallback chain:

1. **Primary** — local GGUF model (`HERA_PRIMARY_OMNI_URL`, default `http://127.0.0.1:8080`)
2. **Secondary** — second local/network node (`HERA_SECONDARY_OMNI_URL`)
3. **Tertiary** — third node (`HERA_TERTIARY_OMNI_URL`)
4. **Cloud** — Gemini / OpenAI-compat fallback. **Only activates if all local/mesh paths fail AND `HERA_ALLOW_CLOUD_FALLBACK` is set.**

Cloud is **failover only**. This is a non-negotiable platform rule (sovereign-first).

---

## Memory & context pipeline (frames C1 + A + E)

When a request arrives, `prepare_runtime_execution_context` (in `ipc/context.rs`) walks this pipeline:

1. **Lightweight check** — trivial greetings short-circuit to a no-tools/no-memory budget.
2. **Difficulty classification** (frame B, in `ipc/difficulty.rs`) — hybrid: a heuristic over length + keyword groups (code/math/reasoning) + density bonus + code fences decides clear cases; gray-zone prompts get a cosine tiebreak against labeled exemplars using the local embedder. Result: Trivial / Normal / Hard → maps to budget mode + `reasoning_effort`.
3. **`build_full_system_prompt`** assembles the prompt in two halves so engines can cache the prefix:

   ```
   stable_prefix  = persona + memento_ctx + CRITICAL RULE + tool schemas + db schema +
                    think/json/language/runtime directives
   dynamic_suffix = recursive_ctx + semantic_ctx (memory that changes per turn)
   ```

   `PromptAssembly` exposes `stable_prefix_chars` and `dynamic_suffix_chars`. Runtime observations persist both to Memento. Live measure on genesis: ~35 KB stable per turn → reused by llama.cpp's KV cache.

   **Context budget calibration (genesis, 2026-06-30):** `Qwen3-30B-A3B-128K` running at `--ctx-size 131072` with YaRN×4 rope scaling (native 32768 → 131K). KV cache quantized q4_0/q4_0 to fit in 48 GB VRAM (2×RTX 3090). System-prompt stable prefix ≈ 8.75K tokens, leaving ~122K for variable context + output. Budget limits in `context_budget_for_mode` are calibrated to this; heavy mode occupies ≤28K tokens total (memory+tools+schema+history), leaving 94K+ for output and growth.

4. **Memory sources** (all from Memento via UDS, all gated by `budget.include_memory && !lightweight`):
   - `fetch_recursive_context(user_id, app_id, session_id)` → Memento `recall_recursive_context` returns the 3-tier scoped_memory state node (project / room / session summaries + working_context + durable_facts + recent_events). Cabling done in frame C1.
   - `fetch_semantic_memories(user_id, app_id, session_id, query)` → embeds the current prompt and asks Memento `semantic_recall` to cosine-rerank scope-filtered rows. Frame A.

5. **Save side**: `handler_generate` / `handler_stream` spawn `save_chat_turn_event` on success, persisting each turn as a scoped_memory event with the embedding attached. Memento's `auto_derive` then derives session/room/project summaries over time.

`canonicalize_user_id(sender_name, chat_id, session_id)` derives a stable user_id from the identifiers Hera already receives (sender_name canonicalized → `chat:<chat_id>` → `anonymous:<session_id>`).

### Embedding model

Default model dir: `/home/paulo/.cache/imagineos-embed-model` (a **stable symlink** to the HF snapshot of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384-dim BERT, strong in Spanish). Override with `HERA_EMBED_MODEL_DIR`. The symlink decouples the code from HF's content-hashed snapshot directory.

If you need to bring the model to a new node:

```bash
# on genesis (source-of-truth):
tar -C /home/paulo/.cache/imagineos-embed-model -hczf /tmp/embed-model.tgz .
scp /tmp/embed-model.tgz <node>:/tmp/
ssh <node> 'mkdir -p ~/.cache/embed-model-multilingual && tar -C ~/.cache/embed-model-multilingual -xzf /tmp/embed-model.tgz && ln -sfn ~/.cache/embed-model-multilingual ~/.cache/imagineos-embed-model'
```

---

## Difficulty routing (frame B)

`difficulty::classify(prompt) -> Difficulty { Trivial, Normal, Hard }`:

- **B1 — predictive escalation of LOCAL effort**: Hard → `context_budget = heavy` (full tools/schema/memory headroom) + `reasoning_effort = high`; Trivial → `lightweight` (no tools/schema/memory, no `<think>` tags). Cloud is never routed-to by difficulty — that violates the sovereign rule.
- **B3 — quality cascade** (`is_low_quality_answer` in `difficulty.rs`): after a non-tool local answer, if it's empty / too short / contains an incapacity disclaimer ("no sé", "no puedo", "no tengo acceso", ...), `handler_generate` escalates **once** to the cloud failover. The escalation only fires if cloud is allowed (`HERA_ALLOW_CLOUD_FALLBACK`) — by default it's NOT, so B3 stays inert and chat stays sovereign.

Run-time signal: log lines `Difficulty-routed query difficulty=… budget=…` (visible in `pm2 logs hera-core`).

---

## Tools (`ai/tool_executor/`)

The dispatcher is split across `dispatch.rs` (per-domain match arms), `registry.rs`, `schema.rs`, `security.rs`, `intent.rs`. Tools are defined as JSON in `../Tools/global/<topic>/` or `../Tools/apps/<app>/`. To add a tool:

1. JSON in `Tools/<scope>/<topic>/<tool_name>.json` (must include `metadata.execution_kind`).
2. Match arm in the appropriate `dispatch.rs::dispatch_*_tool` function calling an executor in `ai/tools/<topic>.rs`.

**Already wired and live (don't re-implement):**

- `save_memory` (`Tools/global/db/save_memory.json` → `productivity::execute_save_memory`) — persists to Memento `save_scoped_memory` with configurable `memory_type` (`task | todo | reminder | note | decision | preference | event | open_loop`). This is the self-editing memory tool: a bot can promote a durable `preference` / `decision` / `fact` from inside a turn without any new code. Frame D.
- `memento_query` — SQL queries against any registered app database via Memento.
- `memento_vector_search` — knowledge search.

The 1500-line file limit applies — split by domain if a file grows.

---

## Build

GPU nodes (genesis, atlas) need `--features local-llm` — without it, the primary engine becomes a stub that fails every request with "Local LLM engine is explicitly disabled", causing Hera to fail over to secondary URLs via WireGuard (10–40 s latency). On 2026-05-21 this caused ~40 s chat latency on paulovila.org; rebuilding genesis Hera with the flag brought latency back to ~1 s.

```bash
# GPU node (local LLM + embeddings + everything)
cd Hera && cargo build --release --bin hera-core --features local-llm

# CPU node that should still have semantic recall (anchor)
cd Hera && cargo build --release --bin hera-core --features embeddings

# CPU node, no embeddings (router only)
cd Hera && cargo build --release --bin hera-core

# Run in IPC mode (production)
IPC_MODE=true ./target/release/hera-core

# Local CLI (routes through Hera IPC)
cargo run -p hera-core --bin claude -- -p "your prompt"
```

**Build pingpong gotcha (genesis disk binary).** Building `--features local-llm` and `--features embeddings` consecutively in the same checkout overwrites `target/release/hera-core` with the last feature set. The running pm2 process keeps the previously-loaded binary in memory, so it doesn't notice — but the **next** restart from pm2 ecosystem would pick up the wrong feature. Pattern:

1. Build local-llm (the production binary for genesis).
2. Restart genesis hera-core if needed.
3. Build embeddings (for anchor) — overwrites `target/release/hera-core` to the embeddings binary.
4. `scp target/release/hera-core` to anchor `~/bin/hera-core` (atomic mv via `.new` because ETXTBSY).
5. **Rebuild local-llm again** to restore genesis's `target/release/hera-core` to its correct feature set.

### Verifying after restart

```bash
pm2 logs hera-core --lines 30 --nostream | grep -E "Sovereign|🧠|disabled"
# Expect "🧠 Sovereign Native Omni Engine mounted!" on a local-llm node.
# Expect "🧠 Sovereign Local LLM disabled via environment flag" on a CPU node — that's correct.
```

---

## Current live capabilities (as of 2026-05-27)

All deployed on genesis (`local-llm`) and anchor (`embeddings`), publication target `paulovilae/hera`:

- **C1 — Recursive scoped-memory wiring** (Hera consumes Memento `recall_recursive_context` + saves each turn via `save_scoped_memory`).
- **A — Semantic recall** (local in-process embeddings via candle BERT, MiniLM-L12 multilingual; Memento `semantic_recall` cosine reranks scope-filtered rows). Available on every node with `embeddings` feature.
- **B — Difficulty routing** (hybrid heuristic + embedding tiebreak; budget + reasoning_effort by class; B3 quality cascade when cloud allowed).
- **C2/C3 — Studio plumbing** (typed artifact dispatch + DuckDB live-binding + Generation DNA + chart) — frontend lives in OS-v3 (`workspace-islands/src/studio.tsx`); Hera serves it via `ai_generate` over IPC.
- **E — Cache-friendly prompt assembly** (stable_prefix + dynamic_suffix order; runtime observations log `stable_prefix_chars` + `dynamic_suffix_chars`).
- **F — Structured outputs pass-through** (`response_format` on ChatRequest).
- **D — Self-editing memory** via the `save_memory` tool with durable `memory_type`.
- **MCP** — `hera_mcp` bin (rmcp-based) under feature `mcp-bridge`.

---

## "Don't rebuild what's already here" rule

Several times during the May 2026 work, a feature looked missing from docs but was already implemented. Before writing new code for a Hera capability, **audit first**:

1. `grep -rn "<feature keyword>" hera-core/src/` — is there code with that name?
2. Are there tool JSONs under `Tools/global/<topic>/` that already do this?
3. Is the IPC action already registered in `ipc/mod.rs`?
4. Is the Memento side already there in `Memento/src/`?

Examples that bit us this session: `scoped_memory` already had the Recursive State Node (C1 just had to cable it); `save_memory` tool already promotes durable facts (D done); `hera_mcp.rs` already exists as a working MCP server; the heritage doc for the Studio said "Not yet ported" but the port was already shipped (`studio.tsx` → `studio.js` built and live).

Treat any doc claim like "Not yet" / "Pending" / "TODO" as **a hypothesis to verify against the code**, not as fact.
