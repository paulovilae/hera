# CLAUDE.md — Hera

This file provides guidance when working inside the `Hera/` submodule.

---

## Submodule Rules

Hera is a git submodule tracked by the parent OS repo. The rules are:

1. **Always commit inside Hera first.** Make changes, `git add`, `git commit` inside `Hera/`. Only then go to the OS root and update the parent pointer (`git add Hera && git commit -m "chore: update Hera pointer"`).
2. **Never modify Hera files and commit only from the OS root.** The parent only tracks a commit hash. If you commit from the OS root without committing inside Hera first, the submodule will show as dirty forever.
3. **`+` prefix in `git submodule status`** means the checked-out commit differs from what the parent records. Resolve by committing inside Hera and updating the parent pointer, not by discarding Hera changes.

Publication target: `paulovilae/hera`

---

## Architecture

```
Hera/
├── hera-core/src/
│   ├── ai/
│   │   ├── mod.rs           — LLMEngine trait, ChatRequest/ChatResponse types
│   │   ├── router.rs        — RouterEngine: primary → secondary → tertiary → cloud fallback
│   │   ├── tool_executor.rs — tool dispatch, all tool implementations
│   │   ├── tools/           — individual tool modules (infra_health, etc.)
│   │   ├── engine_gguf.rs   — local GGUF/llama.cpp engine
│   │   ├── native_engine.rs — native inference engine
│   │   ├── gemini.rs        — Gemini cloud engine
│   │   └── openai_compat.rs — OpenAI-compatible frontend
│   ├── ipc/
│   │   └── runtime_tools.rs — IPC handler, action routing
│   ├── main.rs              — daemon startup, socket listener
│   └── rest_api.rs          — REST API surface (port 3002, fallback only)
└── scripts/
    ├── start_native_omni.sh — starts the local GGUF model server
    └── start_native_draw.sh — starts the image generation backend
```

---

## IPC Socket

Hera listens on `/tmp/hera-core.sock` (Unix Domain Socket). All app communication goes here.

**Request format:**
```json
{
  "action": "generate",
  "payload": {
    "prompt": "...",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 800,
    "temperature": 0.3,
    "permissions": ["garcero"]
  }
}
```

**Tool execution request:**
```json
{
  "action": "execute_tool",
  "payload": {
    "app": "cartera",
    "tool_name": "some_tool",
    "arguments": { "param": "value" }
  }
}
```

**Response:** `{"status": "success", "data": {"result": "..."}}`

---

## Router Engine (router.rs)

The `RouterEngine` implements a four-tier fallback chain:
1. **Primary** — local GGUF model (`HERA_PRIMARY_OMNI_URL`, default `http://127.0.0.1:8080`)
2. **Secondary** — second local/network node (`HERA_SECONDARY_OMNI_URL`)
3. **Tertiary** — third node (`HERA_TERTIARY_OMNI_URL`)
4. **Cloud** — Gemini/OpenAI-compat fallback (only if local path fails AND cloud is allowed)

Cloud fallback is NOT the default. It only activates when all local/mesh paths fail.

---

## Tool Executor (tool_executor.rs)

Tools are defined as JSON in `OS/Tools/` and dispatched here. To add a new tool:
1. Add `<tool_name>.json` to `OS/Tools/global/<topic>/` or `OS/Tools/apps/<app>/`
2. Add a match arm in `tool_executor.rs` with the implementation
3. Keep each tool handler under 80 lines — extract helpers if needed

The 1500-line file limit applies. `tool_executor.rs` is already large; split by domain if it grows further.

---

## Build

**IMPORTANT — GPU nodes must use `--features local-llm`.** Without that flag,
the primary engine becomes a stub that fails every request with "Local LLM
engine is explicitly disabled", causing Hera to failover to secondary URLs
via WireGuard (10-40s latency). On 2026-05-21 this was the root cause of
~40s chat latency on paulovila.org; rebuilding genesis Hera with the flag
brought latency back to ~1s.

```bash
# GPU nodes (genesis, atlas) — required for sovereign-local routing
cd Hera && cargo build --release --bin hera-core --features local-llm

# CPU-only nodes (anchor) — no GPU, uses cloud/secondary path
cd Hera && cargo build --release --bin hera-core

# Run in IPC mode (production)
IPC_MODE=true ./target/release/hera-core

# Local CLI (routes through Hera IPC)
cargo run -p hera-core --bin claude -- -p "your prompt"
cargo run -p hera-core --bin claude    # interactive
```

### Verifying after restart

```bash
pm2 logs hera-core --lines 30 --nostream | grep -E "Sovereign|🧠"
# Expect "🧠 Sovereign Native Omni Engine mounted!" — NOT "disabled"
```

---

## Current State (as of 2026-04-28)

- **9 unstaged files** with substantial changes to router.rs (+329 lines), tool_executor.rs (+163 lines), main.rs, rest_api.rs, infra_health.rs, runtime_tools.rs
- **Parent OS pointer is behind** — Hera is at `82cb66c` but OS root hasn't been updated
- **Memento** is on `codex/serialize-bigdecimal-results` branch — check before depending on Memento features
