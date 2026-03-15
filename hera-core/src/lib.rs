//! Hera - Multimodal LLM Engine
//!
//! Sovereign AI brain responsible for orchestrating SwarmUI models,
//! dynamically assigning GPU pipelines (Instant/Turbo/Quality)
//! based on Universal RBAC capabilities.

pub mod ai;
pub mod ipc_server;
pub mod hardware;
pub mod semantic_object;
pub mod sol;
