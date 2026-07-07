//! Wave 3 — in-flight registry (docs/HERA_OBSERVABILITY_WAVE3_INFLIGHT.md).
//!
//! A process-global map of active generations. Purpose: a long agentic loop
//! (e.g. a self-compile via `hera_compile.sh`, 9 iterations / ~5 min) emits
//! nothing to any observer until it finishes — from outside this is
//! indistinguishable from a hang. This registry lets `hera_inflight` (IPC) and
//! future UX poll "what is Hera doing right now" cheaply (no LLM, no DB).

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

/// Snapshot of one active generation, keyed by trace_id.
#[derive(Debug, Clone)]
pub struct InflightState {
    pub trace_id: String,
    pub app: String,
    pub route: String,
    pub started_at: Instant,
    pub iteration: u32,
    pub max_iters: u32,
    pub current_tool: Option<String>,
    pub last_update: Instant,
    pub node: String,
}

fn registry() -> &'static Mutex<HashMap<String, InflightState>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, InflightState>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lock the registry, recovering from poisoning — a panicked holder must never
/// wedge the mutex for every future request.
fn lock() -> MutexGuard<'static, HashMap<String, InflightState>> {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Register a new in-flight generation. No-op if `trace_id` is empty — an
/// unattributed request would pollute the registry with a row nothing could
/// ever uniquely `remove`.
pub fn insert(trace_id: &str, app: &str, route: &str, node: &str) {
    if trace_id.is_empty() {
        return;
    }
    let now = Instant::now();
    lock().insert(
        trace_id.to_string(),
        InflightState {
            trace_id: trace_id.to_string(),
            app: app.to_string(),
            route: route.to_string(),
            started_at: now,
            iteration: 0,
            max_iters: 0,
            current_tool: None,
            last_update: now,
            node: node.to_string(),
        },
    );
}

/// Update the agentic-loop iteration counters for a trace_id. No-op if the
/// trace_id isn't registered (e.g. called after `remove`, or never inserted).
pub fn set_iteration(trace_id: &str, iteration: u32, max_iters: u32) {
    if trace_id.is_empty() {
        return;
    }
    if let Some(state) = lock().get_mut(trace_id) {
        state.iteration = iteration;
        state.max_iters = max_iters;
        state.last_update = Instant::now();
    }
}

/// Update the tool currently being dispatched for a trace_id.
pub fn set_tool(trace_id: &str, tool: Option<&str>) {
    if trace_id.is_empty() {
        return;
    }
    if let Some(state) = lock().get_mut(trace_id) {
        state.current_tool = tool.map(|t| t.to_string());
        state.last_update = Instant::now();
    }
}

/// Remove a trace_id from the registry on terminal state (done/error/empty/max_iters).
pub fn remove(trace_id: &str) {
    if trace_id.is_empty() {
        return;
    }
    lock().remove(trace_id);
}

/// Serialize the current registry for the `hera_inflight` IPC action.
/// `elapsed_ms` / `last_update_ms_ago` are computed at snapshot time since
/// `Instant` itself isn't serializable.
pub fn snapshot() -> Vec<serde_json::Value> {
    let now = Instant::now();
    lock()
        .values()
        .map(|s| {
            serde_json::json!({
                "trace_id": s.trace_id,
                "app": s.app,
                "route": s.route,
                "elapsed_ms": now.duration_since(s.started_at).as_millis() as u64,
                "iteration": s.iteration,
                "max_iters": s.max_iters,
                "current_tool": s.current_tool,
                "last_update_ms_ago": now.duration_since(s.last_update).as_millis() as u64,
                "node": s.node,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_set_snapshot_remove_roundtrip() {
        insert("wave3-t1", "capacita", "coding", "genesis");
        set_iteration("wave3-t1", 3, 25);
        set_tool("wave3-t1", Some("cargo_check"));

        let snap = snapshot();
        let row = snap
            .iter()
            .find(|v| v["trace_id"] == "wave3-t1")
            .expect("row present after insert");
        assert_eq!(row["app"], "capacita");
        assert_eq!(row["route"], "coding");
        assert_eq!(row["iteration"], 3);
        assert_eq!(row["max_iters"], 25);
        assert_eq!(row["current_tool"], "cargo_check");
        assert_eq!(row["node"], "genesis");

        remove("wave3-t1");
        let snap2 = snapshot();
        assert!(snap2.iter().all(|v| v["trace_id"] != "wave3-t1"));
    }

    #[test]
    fn empty_trace_id_is_noop_everywhere() {
        insert("", "app", "route", "node");
        set_iteration("", 1, 2);
        set_tool("", Some("x"));
        assert!(snapshot().iter().all(|v| v["trace_id"] != ""));
        remove(""); // must not panic
    }

    #[test]
    fn set_iteration_on_unknown_trace_id_is_noop() {
        // Does not panic, does not insert a new row.
        set_iteration("never-inserted", 5, 10);
        assert!(
            snapshot()
                .iter()
                .all(|v| v["trace_id"] != "never-inserted")
        );
    }
}
