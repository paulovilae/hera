# Hera Capability Refactor Plan

## Purpose

This document defines the first safe refactor path for Hera:

1. make capabilities explicit
2. let Hera report what is compiled vs runtime-enabled
3. preserve current default behavior
4. prepare a later split of the heavy dependency graph into modular crates

This phase does **not** attempt a risky monolithic rewrite.

## Current Problem

`hera-core` currently mixes three concerns:

- orchestration and IPC
- heavy local inference stacks
- execution/tool subsystems with large transitive graphs

The result is that small binaries like `hera_mcp` still compile against a large graph because they share the `hera-core` library surface.

The biggest weight centers are:

- `candle-core`, `candle-nn`, `candle-transformers`, `llama-cpp-2`, `tokenizers`, `hf-hub`
- `hera-execution -> lancedb -> lance -> datafusion -> arrow`
- standard server/runtime crates like `tokio`, `axum`, and `reqwest`

## Phase 1: Capability Model

Implemented in code:

- `local_llm`
- `vision`
- `audio_tts`
- `audio_stt`
- `desktop_control`
- `execution_tools`
- `mcp_bridge`

Each capability should answer:

- is it compiled?
- is it runtime-enabled?
- what is its startup cost?
- what is it for?

## Phase 2: Explicit Cargo Features

### `hera-core`

Feature groups:

- `local-llm`
- `vision`
- `audio`
- `desktop-control`
- `execution-tools`
- `mcp-bridge`

Current default behavior stays unchanged by keeping the runtime features in `default`.

### `hera-execution`

Feature groups:

- `vector`
- `market`
- `docs`
- `web`
- `agents`
- `workflow`

Implemented in this phase:

- `vector` now gates LanceDB and Arrow dependencies plus the `memory` module
- `docs` now gates PDF/XLS extraction dependencies plus native document tools
- `web` now gates HTTP/MCP client dependencies and web-backed execution paths
- `market` now gates Yahoo Finance integration
- `workflow` remains the orchestration surface, but degrades gracefully when a required domain feature is disabled

Verified build matrix:

- default build
- `--no-default-features`
- `--no-default-features --features docs`
- `--no-default-features --features web,agents,workflow`

## Phase 3: Runtime Awareness

At startup Hera should build a capability registry and log:

- compiled status
- env-enabled status
- startup cost
- purpose note

The runtime should check the registry instead of hardcoding env checks everywhere.

Examples:

- `HERA_ENABLE_LLM`
- `HERA_ENABLE_PARLER`
- `HERA_ENABLE_WHISPER`

## Phase 4: Heavy Graph Decomposition

This is the real dependency reduction phase.

### Target split

- `hera-core`
  - IPC
  - routing
  - persona loading
  - tool dispatch
  - capability registry
- `hera-web`
  - MCP HTTP client
  - web search / scrape
  - image / audio / video web-facing orchestration
- `hera-execution`
  - tool execution facade
  - workflow routing
  - thin re-exports over domain crates
- `hera-vector`
  - LanceDB / vector / retrieval
- `hera-docs`
  - PDF / XLS / OCR / extraction
- `hera-market`
  - finance / quote fetch helpers
- later: `hera-inference`
  - local LLM
  - vision
  - audio
- later: `hera-desktop`
  - desktop input automation
- later: `hera-mcp`
  - MCP adapters only

### Immediate rule

`hera_mcp` should stay a thin adapter and must not become a second execution engine.

### Implemented split

This phase extracted the first heavy execution slices into standalone workspace crates:

- `hera-web`
- `hera-docs`
- `hera-vector`
- `hera-market`

Current consequences:

- `hera-core` now depends on `hera-web` directly for the Hera web/multimodal wrapper
- `hera-execution` now acts as a facade over those domain crates
- Lance/DataFusion weight is isolated under `hera-vector`
- PDF/XLS extraction weight is isolated under `hera-docs`
- Yahoo Finance weight is isolated under `hera-market`

What still remains heavy:

- `hera-core` still owns the Candle / llama / inference graph directly
- minimal `hera_mcp` builds still pay for `hera-core` inference dependencies until those move into a dedicated inference crate

## Recommended Migration Order

1. keep current defaults
2. keep current runtime behavior
3. move capability detection into one place
4. make new binaries depend on thinner interfaces
5. split `hera-execution` feature domains
6. move `lancedb` behind a vector-specific crate or feature
7. move Candle / llama stacks behind inference-specific crate boundaries
8. move local inference engines into `hera-inference`
9. make `hera_mcp` depend on thin IPC/tool surfaces instead of `hera-core` monolith

## High-Value Next Steps

### Step A

Create:

- `src/capabilities/mod.rs`
- `src/capabilities/registry.rs`
- `src/capabilities/types.rs`

The current phase started this inside a single module; later it can be split cleanly.

### Step B

Add an execution manifest layer in `hera-execution`:

- tool name
- required capability
- warmup policy
- permission scope

This lets Hera ask "what capability does this request need?" before touching the subsystem.

### Step C

Split `hera_mcp` into its own package once a thin IPC/tool facade exists.

That is the cleanest way to stop MCP binaries from pulling the full inference graph.

### Step D

Move the now-feature-gated `vector`, `docs`, `market`, and `web` slices into separate crates.

That is the point where Cargo can stop compiling their source trees entirely for binaries that do not depend on them.

## Non-Goals For This Phase

- no behavior change to default Hera startup
- no removal of existing engines
- no forced crate split yet
- no breaking changes to current IPC payloads

## Definition Of Done For Phase 1

- Hera has explicit feature groups in Cargo
- Hera reports capability status at startup
- runtime checks use the capability registry for core toggles
- MCP is feature-gated as an optional adapter
- `hera-execution` selective builds compile without dragging the full dependency graph
- the refactor path is documented so later work is mechanical
