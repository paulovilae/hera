# Hera

**Sovereign, local-first LLM orchestration engine written in Rust.**

[![Rust](https://img.shields.io/badge/rust-2024--edition-orange)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Status: Production](https://img.shields.io/badge/status-production-green)]()

Hera is a daemon that runs on bare metal and handles all LLM inference, tool execution, and multi-agent orchestration for a multi-app platform — with no mandatory cloud dependency. It listens on a Unix Domain Socket, speaks JSON, and routes requests through a 4-tier local→cloud fallback chain where cloud only activates if you opt in.

Deployed on genesis (2× RTX 3090, Qwen3-30B-A3B) and anchor (GCP edge, CPU + embeddings). Powers five production apps simultaneously across web, Telegram, and WhatsApp.

---

## What it does

- **Routes LLM requests** through a 4-tier chain: primary local model → secondary node → tertiary mesh → cloud (opt-in only)
- **Executes tools** — SQL queries, file edits, shell commands, cargo builds, web fetches, image generation, transcription
- **Runs an agentic loop** — multi-turn act→observe→fix cycles with 6 coding tools (`cargo_check`, `cargo_test`, `edit_file`, `write_file`, `grep_search`, `glob_search`)
- **Manages memory** — integrates with Memento for scoped memory, semantic recall, and per-user persistent context
- **Classifies query difficulty** — hybrid heuristic + BERT embedding tiebreak → Trivial/Normal/Hard → maps to budget mode + reasoning effort
- **Assembles cache-friendly prompts** — stable prefix (persona + memory + schemas, ~35KB) + dynamic suffix (per-turn context), reused by llama.cpp's KV cache
- **Serves as an MCP server** — `hera_mcp` binary (feature `mcp-bridge`) exposes Hera over the Model Context Protocol

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Hera Daemon                │
                    │         /tmp/hera-core.sock             │
                    │                                         │
  App / Bot ──────► │  IPC Dispatcher (action → handler)      │
  (JSON over UDS)   │                                         │
                    │  ┌─────────────┐  ┌──────────────────┐ │
                    │  │  Generator  │  │  Tool Executor   │ │
                    │  │             │  │                  │ │
                    │  │  Router     │  │  global tools    │ │
                    │  │  ├─primary  │  │  app tools       │ │
                    │  │  ├─secondary│  │  coding tools    │ │
                    │  │  ├─tertiary │  │  (agentic loop)  │ │
                    │  │  └─cloud*   │  └──────────────────┘ │
                    │  └─────────────┘                        │
                    │                                         │
                    │  ┌─────────────┐  ┌──────────────────┐ │
                    │  │  Embeddings │  │  Memory (IPC)    │ │
                    │  │  candle BERT│  │  → Memento UDS   │ │
                    │  │  MiniLM-L12 │  │  scoped memory   │ │
                    │  │  384-dim    │  │  semantic recall │ │
                    │  └─────────────┘  └──────────────────┘ │
                    └─────────────────────────────────────────┘

  * cloud only activates if HERA_ALLOW_CLOUD_FALLBACK is set
```

---

## Part of the Vilaros OS stack

Hera is the reasoning core. The full platform has five shared services:

| Service | Role |
|---|---|
| **Hera** (this repo) | LLM orchestration, tool execution, multi-agent |
| **Memento** | Persistent memory, semantic recall, scoped context |
| **Sentinel** | Ingress, TLS termination, identity-first edge |
| **Argus** | Hardware detection, cluster service placement |
| **OS-v3** | Governance, app registry, shared UI, SDK |

Each service communicates over Unix Domain Sockets with JSON. No service calls the LLM directly — everything goes through Hera.

---

## Cargo features

| Feature | Purpose | Use on |
|---|---|---|
| `local-llm` | GGUF/llama.cpp engine + CUDA + embeddings | GPU nodes |
| `embeddings` | candle BERT only (CPU, no llama.cpp) | CPU edge nodes |
| `mcp-bridge` | `hera_mcp` binary (rmcp MCP server) | MCP clients |

---

## Build

```bash
# GPU node — local LLM + CUDA + embeddings (production)
cargo build --release --bin hera-core --features local-llm

# CPU edge node — semantic recall only, no local LLM
cargo build --release --bin hera-core --features embeddings

# MCP server binary
cargo build --release --bin hera_mcp --features mcp-bridge

# Run as daemon
IPC_MODE=true ./target/release/hera-core

# Interactive CLI (routes through the running daemon)
cargo run -p hera-core --bin claude

# Single prompt
cargo run -p hera-core --bin claude -- -p "your prompt"
```

Verify after restart:
```bash
pm2 logs hera-core --lines 30 --nostream | grep -E "Sovereign|🧠|disabled"
# GPU node: "🧠 Sovereign Native Omni Engine mounted!"
# CPU node: "🧠 Sovereign Local LLM disabled via environment flag"
```

---

## IPC protocol

All communication is JSON over Unix Domain Socket at `/tmp/hera-core.sock`.

**Generate (non-streaming):**
```json
{
  "action": "generate",
  "payload": {
    "prompt": "Summarize this document",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 800,
    "temperature": 0.3,
    "app": "myapp",
    "route_profile": "standard",
    "session_id": "abc123",
    "permissions": ["memento_query"]
  }
}
```

**Streaming:**
```json
{ "action": "generate_stream", "payload": { ... } }
```

**Direct tool execution:**
```json
{
  "action": "execute_tool",
  "payload": {
    "tool_name": "memento_query",
    "arguments": { "app": "myapp", "query": "SELECT * FROM items LIMIT 5" }
  }
}
```

**Other actions:** `embed`, `delegate_task`, `route_health`, `generate_image`, `vision_analysis`, `transcribe_audio`, `execute_dag`, `get_tools`

---

## Route profiles

Route profiles map a request identifier to a context budget mode:

| Mode | Tools | DB schema | Memory | Use for |
|---|---|---|---|---|
| `lightweight` | ✗ | ✗ | ✗ | Greetings, simple Q&A |
| `standard` | ✓ | ✓ | ✓ | Data-querying bots |
| `heavy` | ✓ | ✓ | ✓ (full) | Multi-step reasoning |

---

## Key environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HERA_PRIMARY_OMNI_URL` | `http://127.0.0.1:8080` | Primary local model server |
| `HERA_SECONDARY_OMNI_URL` | — | Secondary node (mesh) |
| `HERA_TERTIARY_OMNI_URL` | — | Tertiary node |
| `HERA_ALLOW_CLOUD_FALLBACK` | unset | Enable cloud fallover (unset = sovereign only) |
| `HERA_AGENTIC_LOOP` | unset | Enable multi-turn tool loop (`1` to enable) |
| `HERA_EMBED_MODEL_DIR` | `~/.cache/imagineos-embed-model` | Path to multilingual BERT model |
| `HERA_CLOUD_MAX_CALLS_PER_WINDOW` | — | Cloud rate limit |
| `HERA_CLOUD_MAX_TOKENS_PER_DAY` | — | Cloud daily token budget |
| `HERA_ALLOW_PAID_CLOUD_MODELS` | unset | Allow non-free cloud models |
| `OS_V3_INTERNAL_URL` | `http://127.0.0.1:5177` | OS-v3 governance service |
| `OS_ADMIN_EMAIL` | `admin@localhost` | Admin identity for magic links |
| `OS_ADMIN_NAME` | `Admin` | Admin display name |
| `IPC_MODE` | unset | Run as UDS daemon (`true`) vs REST |

---

## Embedding model

Hera uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual, strong in Spanish) for difficulty classification tiebreaks and semantic recall. Runs CPU-only via candle — no GPU required for the `embeddings` feature.

```bash
# Set up model dir (symlink to HF snapshot)
export HERA_EMBED_MODEL_DIR=~/.cache/imagineos-embed-model
```

---

## Agentic loop

When `HERA_AGENTIC_LOOP=1`, Hera runs multi-turn act→observe→fix cycles instead of single-shot generation. Available tools for the coding context:

| Tool | What it does |
|---|---|
| `edit_file` | Surgical string replacement in a file |
| `write_file` | Write/overwrite a file |
| `grep_search` | Regex search across files |
| `glob_search` | Find files by pattern |
| `cargo_check` | Run `cargo check` and return diagnostics |
| `cargo_test` | Run `cargo test` and return results |

Enabled by `route_profile: "coding"` or `caller: "coding"`. Low deterministic temperature (0.2) applies only to coding contexts.

---

## Memory pipeline

Per request, `prepare_runtime_execution_context` runs:

1. Classify difficulty (heuristic + embedding tiebreak)
2. Fetch recursive scoped memory from Memento (`recall_recursive_context` — project/room/session summaries + durable facts)
3. Fetch semantic memories (BERT embed query → cosine rerank)
4. Assemble prompt: `stable_prefix` (cacheable, ~35KB) + `dynamic_suffix` (per-turn)
5. After response: persist turn to Memento (`save_scoped_memory`)

---

## Cost-safety gates (cloud)

Three independent gates at the cloud chokepoint:

1. **Rate limit + daily budget** — `HERA_CLOUD_MAX_CALLS_PER_WINDOW` + `HERA_CLOUD_MAX_TOKENS_PER_DAY`
2. **Free-tier enforcement** — rejects non-`:free` OpenRouter models unless `HERA_ALLOW_PAID_CLOUD_MODELS` is set
3. **App-level never-external policy** — `app_cloud_policy.yaml` in OS-v3 hardcodes certain apps as never-external regardless of workload

---

## License

MIT — see [LICENSE](LICENSE)

---

*Hera is the reasoning core of [Vilaros OS](https://vilaros.ai) — a sovereign AI operating system built in Rust.*
