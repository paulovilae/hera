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
