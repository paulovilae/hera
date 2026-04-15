# 👁️ Hera Core (Multimodal LLM Engine)

**Role:** The Sovereign AI Executor
**Stack:** Pure Rust (Candle Framework / Llama.cpp)
**Network Status:** Headless IPC Daemon (Portless)

## Bundle Position

`Hera Core` is not the full Ava assistant by itself.

It is one component inside the canonical Ava bundle:

- `Argus`
- `Sentinel`
- `Imaginclaw`
- `Hera/hera-core`
- `Hera/diakonos-core`
- `Memento`

Before diagnosing assistant capability or adding features, read:

- [Ava Bundle Capabilities Matrix](/home/paulo/Programs/apps/OS/docs/AVA_BUNDLE_CAPABILITIES_MATRIX.md)

Mandatory rule:

- Do not treat missing orchestration, approvals, channel handling, memory UX, or task UX as missing `Hera` execution capability without checking the full bundle first.
- Do not duplicate memory, edge, or orchestration behavior inside `Hera` when those belong to `Memento`, `Sentinel`, or `Imaginclaw`.

## Characteristics
Hera is the pure computational brain of the Vilaros ecosystem. Stripped of all web server and UI overhead, Hera runs as a highly optimized, headless background daemon.

- **Pure Speed Architecture**: Communicates exclusively with Vilaros OS via zero-latency Unix Domain Sockets (UDS) / IPC.
- **Persistent GPU Memory**: Handled independently by PM2. If the OS interface crashes or refreshes, Hera's massive tensor matrices stay securely loaded in VRAM, eliminating expensive AI reload times.
- **No HTTP Exposure**: It does not listen on any public or localhost HTTP ports, rendering it utterly invisible to outside network scans.

## Implementation Plan
1. **Remove HTTP Frameworks**: Strip Axum/Actix from the `hera-core` crate.
2. **Socket Listener**: Implement a fast `tokio::net::UnixListener` (or named pipes) to accept raw binary or fast JSON-RPC payloads strictly from the OS gateway.
3. **Continuous Inference Loop**: Optimize the Candele/Model generation loops to stream tokens directly back through the open IPC socket with zero overhead padding.

## Local Claude-Style CLI

`hera-core` now includes a local `claude` terminal client that routes through Hera IPC instead of talking to Anthropic directly.

Run it from the workspace:

```bash
cargo run -p hera-core --bin claude -- -p "Summarize this repo"
```

Interactive mode:

```bash
cargo run -p hera-core --bin claude
```

Install a global `claude` command:

```bash
./Hera/hera-core/scripts/install_claude_cli.sh
```

If you want the real Claude Code client pointed at Hera, install the dedicated helper instead:

```bash
./Hera/hera-core/scripts/install_claude_code_hera.sh
```

If you installed the native `claude` client, add a launcher for it too:

```bash
./Hera/hera-core/scripts/install_claude_hera.sh
```

Notes:

- default socket: `/tmp/hera-core.sock`
- default mode: streaming
- use `--no-stream` for a full buffered response
- use `--permission <tool>` to allow specific Hera tools during a request

## Claude-Compatible API

`hera-core` now also exposes a native Anthropic/Claude-compatible frontend on the REST server so external tools can talk to Hera without a separate proxy layer.

Available endpoints:

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /v1/models`

Point Claude Code at Hera with environment variables like:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:3002
export ANTHROPIC_API_KEY=dummy
```

Notes:

- requests are translated into Hera's internal chat request format
- streaming responses are served as SSE
- model routing still happens inside Hera
- `claude-code-hera` is a thin launcher around the real Claude Code client:

```bash
claude-code-hera -p --model hera-local-model "Reply with exactly: hera ok"
```

- `claude-hera` wraps the installed native `claude` binary with Hera env vars:

```bash
claude-hera --bare --model hera-local-model
```
