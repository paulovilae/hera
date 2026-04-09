//! Legacy re-export shim — delegates to the modular `ipc/` subsystem.
//!
//! This file exists only for backward compatibility with `main.rs` and
//! any external crate code that imports `hera_core::ipc_server::*`.
//!
//! All business logic now lives under `src/ipc/`.

pub use crate::ipc::{serve, IpcPayload, IpcResponse, IpcState};
