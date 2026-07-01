//! Hera Watchdog — Autonomous Emergency Detection & Self-Healing
//!
//! A background `tokio` task that runs every 60 seconds. It:
//! 1. Checks all PM2 services for crashes/restarts
//! 2. Checks disk space, VRAM, and WireGuard tunnel
//! 3. Auto-restarts crashed services (up to 3 retries per cycle)
//! 4. Logs all actions to a persistent watchdog log
//! 5. Writes emergency alerts to a notification file that messaging
//!    channels (Telegram/WhatsApp/Imaginclaw) can poll
//!
//! The watchdog is "fire and forget" — it heals silently in the background
//! and only escalates to the user when it can't fix something.

use serde_json::Value;
use std::collections::HashMap;
use tracing::{error, info, warn};

/// Path where the watchdog writes emergency alerts for external consumers
const ALERT_FILE: &str = "/tmp/hera-watchdog-alerts.jsonl";
/// Path for the persistent watchdog action log
const WATCHDOG_LOG: &str = "/home/paulo/Programs/apps/OS/Apps/OS-v3/storage/logs/watchdog.jsonl";
/// How often the watchdog ticks (seconds)
const TICK_INTERVAL_SECS: u64 = 60;
/// Max consecutive auto-restart attempts per service before escalating
const MAX_AUTO_RESTARTS: u32 = 3;

/// In-memory state for tracking restart attempts per service
struct WatchdogState {
    restart_counts: HashMap<String, u32>,
    /// Last restart count at which a PM2_HIGH_RESTARTS warning was emitted per service.
    /// Only warn again once restarts have grown by at least HIGH_RESTART_WARN_STEP.
    high_restart_last_warned: HashMap<String, u64>,
    last_tick: std::time::Instant,
}

const HIGH_RESTART_WARN_STEP: u64 = 20;

impl WatchdogState {
    fn new() -> Self {
        Self {
            restart_counts: HashMap::new(),
            high_restart_last_warned: HashMap::new(),
            last_tick: std::time::Instant::now(),
        }
    }

    /// Reset restart counts every 10 minutes to allow retries after cooldown
    fn maybe_reset_counts(&mut self) {
        if self.last_tick.elapsed().as_secs() > 600 {
            self.restart_counts.clear();
            self.last_tick = std::time::Instant::now();
        }
    }
}

/// Spawn the watchdog as a background tokio task.
/// Call this once from `main.rs` after all engines are initialized.
pub fn spawn_watchdog() {
    tokio::spawn(async move {
        info!(
            "🐕 [Watchdog] Autonomous emergency watchdog started (tick: {}s)",
            TICK_INTERVAL_SECS
        );
        let mut state = WatchdogState::new();

        // Initial delay — let all services finish booting
        tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;

        loop {
            state.maybe_reset_counts();

            if let Err(e) = watchdog_tick(&mut state).await {
                error!("🐕 [Watchdog] Tick error: {}", e);
            }

            tokio::time::sleep(tokio::time::Duration::from_secs(TICK_INTERVAL_SECS)).await;
        }
    });
}

/// One full watchdog cycle
async fn watchdog_tick(state: &mut WatchdogState) -> Result<(), String> {
    let mut emergencies: Vec<Emergency> = Vec::new();

    // ── 1. Check PM2 services ──
    check_pm2_services(state, &mut emergencies);

    // ── 2. Check disk space ──
    check_disk_space(&mut emergencies);

    // ── 3. Check VRAM ──
    check_vram(&mut emergencies);

    // ── 4. Check WireGuard ──
    check_wireguard(&mut emergencies);

    // ── 5. Check Sentinel (port 3000) ──
    check_sentinel(state, &mut emergencies);

    // ── 6. Purge expired audit_log rows ──
    purge_audit_log().await;

    // ── 6. Process emergencies ──
    for emergency in &emergencies {
        log_watchdog_event(emergency);

        match emergency.severity {
            Severity::AutoHealed => {
                info!("🐕 [Watchdog] Auto-healed: {}", emergency.message);
            }
            Severity::Warning => {
                warn!("🐕 [Watchdog] Warning: {}", emergency.message);
            }
            Severity::Critical => {
                error!(
                    "🐕 [Watchdog] ⚠️ CRITICAL (needs human): {}",
                    emergency.message
                );
                write_alert(emergency);
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
enum Severity {
    AutoHealed, // Fixed automatically, just logged
    Warning,    // Something degraded but not critical
    Critical,   // Can't auto-fix, escalate to user
}

#[derive(Debug, Clone)]
struct Emergency {
    severity: Severity,
    category: String,
    service: String,
    message: String,
    action_taken: String,
}

/// Check all PM2 services and auto-restart crashed ones
fn check_pm2_services(state: &mut WatchdogState, emergencies: &mut Vec<Emergency>) {
    let output = match std::process::Command::new("pm2").arg("jlist").output() {
        Ok(o) if o.status.success() => o,
        _ => return,
    };

    let out_str = String::from_utf8_lossy(&output.stdout);
    let procs: Vec<Value> = match serde_json::from_str(&out_str) {
        Ok(p) => p,
        Err(_) => return,
    };

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    for proc in &procs {
        let name = proc.get("name").and_then(|n| n.as_str()).unwrap_or("?");
        let env = proc.get("pm2_env");
        let status = env
            .and_then(|e| e.get("status"))
            .and_then(|s| s.as_str())
            .unwrap_or("?");
        let restarts = env
            .and_then(|e| e.get("restart_time"))
            .and_then(|r| r.as_u64())
            .unwrap_or(0);
        // Read pm2's own restart_delay so we don't compound a rotation that
        // pm2 is already pacing. If pm2 set a delay (e.g. 8000 ms for movilo
        // to let port 5175 release), the watchdog must give pm2 that window
        // before issuing its own restart — otherwise we race pm2's retry and
        // inflate the restart counter.
        let restart_delay_ms = env
            .and_then(|e| e.get("restart_delay"))
            .and_then(|r| r.as_u64())
            .unwrap_or(0);
        let pm_uptime = env
            .and_then(|e| e.get("pm_uptime"))
            .and_then(|u| u.as_u64())
            .unwrap_or(0);
        let since_last_start_ms = now_ms.saturating_sub(pm_uptime);

        // Detect crashed/errored/stopped services
        if status == "errored" || status == "stopped" {
            // Respect pm2's restart_delay: if pm2 just stopped the process and
            // is about to restart it itself, skip this tick.
            if restart_delay_ms > 0 && since_last_start_ms < restart_delay_ms + 2000 {
                continue;
            }
            let attempt_count = state.restart_counts.entry(name.to_string()).or_insert(0);

            if *attempt_count < MAX_AUTO_RESTARTS {
                // Attempt auto-restart
                let restart_result = std::process::Command::new("pm2")
                    .args(["restart", name])
                    .output();

                *attempt_count += 1;

                match restart_result {
                    Ok(r) if r.status.success() => {
                        emergencies.push(Emergency {
                            severity: Severity::AutoHealed,
                            category: "PM2_RESTART".to_string(),
                            service: name.to_string(),
                            message: format!(
                                "Service '{}' was {} — auto-restarted (attempt {}/{})",
                                name, status, attempt_count, MAX_AUTO_RESTARTS
                            ),
                            action_taken: format!("pm2 restart {}", name),
                        });
                    }
                    _ => {
                        emergencies.push(Emergency {
                            severity: Severity::Critical,
                            category: "PM2_RESTART_FAILED".to_string(),
                            service: name.to_string(),
                            message: format!(
                                "Service '{}' is {} — auto-restart FAILED (attempt {}/{})",
                                name, status, attempt_count, MAX_AUTO_RESTARTS
                            ),
                            action_taken: "pm2 restart failed".to_string(),
                        });
                    }
                }
            } else {
                // Exhausted retries — escalate to human
                emergencies.push(Emergency {
                    severity: Severity::Critical,
                    category: "PM2_CRASH_LOOP".to_string(),
                    service: name.to_string(),
                    message: format!(
                        "Service '{}' is {} after {} auto-restart attempts — NEEDS HUMAN INTERVENTION. Check logs: pm2 logs {} --err --lines 30",
                        name, status, MAX_AUTO_RESTARTS, name
                    ),
                    action_taken: "exhausted auto-restart retries".to_string(),
                });
            }
        }

        // Detect crash loops even for "online" services — but only log once per
        // HIGH_RESTART_WARN_STEP increment to avoid flooding the watchdog log.
        if status == "online" && restarts > 20 {
            let last = state
                .high_restart_last_warned
                .get(name)
                .copied()
                .unwrap_or(0);
            let bucket = (restarts / HIGH_RESTART_WARN_STEP) * HIGH_RESTART_WARN_STEP;
            if bucket > last {
                state
                    .high_restart_last_warned
                    .insert(name.to_string(), bucket);
                emergencies.push(Emergency {
                    severity: Severity::Warning,
                    category: "PM2_HIGH_RESTARTS".to_string(),
                    service: name.to_string(),
                    message: format!(
                        "Service '{}' is online but has {} restarts — possible instability",
                        name, restarts
                    ),
                    action_taken: "logged warning".to_string(),
                });
            }
        }
    }
}

/// Check disk space — services crash silently when disk is full
fn check_disk_space(emergencies: &mut Vec<Emergency>) {
    let output = match std::process::Command::new("df")
        .args(["-h", "--output=target,pcent,avail", "/", "/home"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return,
    };

    let out_str = String::from_utf8_lossy(&output.stdout);
    for line in out_str.lines().skip(1) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let mount = parts[0];
            let pct_str = parts[1].trim_end_matches('%');
            if let Ok(pct) = pct_str.parse::<u32>() {
                if pct > 95 {
                    emergencies.push(Emergency {
                        severity: Severity::Critical,
                        category: "DISK_CRITICAL".to_string(),
                        service: mount.to_string(),
                        message: format!(
                            "DISK CRITICAL: {} is at {}% (only {} free) — services WILL crash on write",
                            mount, pct, parts[2]
                        ),
                        action_taken: "escalated to user".to_string(),
                    });
                } else if pct > 90 {
                    emergencies.push(Emergency {
                        severity: Severity::Warning,
                        category: "DISK_WARNING".to_string(),
                        service: mount.to_string(),
                        message: format!(
                            "Disk warning: {} is at {}% (only {} free)",
                            mount, pct, parts[2]
                        ),
                        action_taken: "logged warning".to_string(),
                    });
                }
            }
        }
    }
}

/// Check GPU VRAM exhaustion
fn check_vram(emergencies: &mut Vec<Emergency>) {
    let output = match std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return,
    };

    let out_str = String::from_utf8_lossy(&output.stdout);
    for line in out_str.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 3 {
            let idx = parts[0].trim();
            let used: f64 = parts[1].trim().parse().unwrap_or(0.0);
            let total: f64 = parts[2].trim().parse().unwrap_or(1.0);
            let pct = (used / total) * 100.0;
            if pct > 98.0 {
                emergencies.push(Emergency {
                    severity: Severity::Critical,
                    category: "VRAM_EXHAUSTED".to_string(),
                    service: format!("GPU-{}", idx),
                    message: format!(
                        "GPU {} VRAM exhausted: {:.0}MB / {:.0}MB ({:.0}%) — new AI loads WILL fail",
                        idx, used, total, pct
                    ),
                    action_taken: "escalated to user".to_string(),
                });
            }
        }
    }
}

/// Check WireGuard tunnel health
fn check_wireguard(emergencies: &mut Vec<Emergency>) {
    let output = match std::process::Command::new("wg").arg("show").output() {
        Ok(o) if o.status.success() => o,
        _ => return, // wg not installed or no permission — skip silently
    };

    let out_str = String::from_utf8_lossy(&output.stdout);
    if out_str.trim().is_empty() {
        emergencies.push(Emergency {
            severity: Severity::Critical,
            category: "WIREGUARD_DOWN".to_string(),
            service: "wg0".to_string(),
            message: "WireGuard tunnel is DOWN — external traffic cannot reach services"
                .to_string(),
            action_taken: "escalated to user (requires sudo)".to_string(),
        });
    }
}

/// Check Sentinel reverse proxy (port 3000) — the gateway for all traffic
fn check_sentinel(state: &mut WatchdogState, emergencies: &mut Vec<Emergency>) {
    // Quick TCP probe to port 3000
    let probe = std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--connect-timeout",
            "2",
            "http://127.0.0.1:3000/",
        ])
        .output();

    match probe {
        Ok(output) => {
            let code_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let code: u16 = code_str.parse().unwrap_or(0);
            if code == 0 {
                let attempt_count = state
                    .restart_counts
                    .entry("sentinel".to_string())
                    .or_insert(0);
                if *attempt_count < MAX_AUTO_RESTARTS {
                    let _ = std::process::Command::new("pm2")
                        .args(["restart", "sentinel"])
                        .output();
                    *attempt_count += 1;
                    emergencies.push(Emergency {
                        severity: Severity::AutoHealed,
                        category: "SENTINEL_RESTART".to_string(),
                        service: "sentinel".to_string(),
                        message: format!(
                            "Sentinel (port 3000) was unreachable — auto-restarted (attempt {}/{})",
                            attempt_count, MAX_AUTO_RESTARTS
                        ),
                        action_taken: "pm2 restart sentinel".to_string(),
                    });
                } else {
                    emergencies.push(Emergency {
                        severity: Severity::Critical,
                        category: "SENTINEL_DOWN".to_string(),
                        service: "sentinel".to_string(),
                        message: "Sentinel (port 3000) is DOWN and auto-restart exhausted — ALL external traffic is BLOCKED".to_string(),
                        action_taken: "exhausted auto-restart retries".to_string(),
                    });
                }
            }
        }
        Err(_) => {} // curl not available, skip
    }
}

/// Purge expired audit_log rows.
/// - Sentinel ingress traces (high volume): expire after 7 days
/// - All other rows: expire after 30 days
/// - Rows with explicit retention_until: expire when that timestamp passes
async fn purge_audit_log() {
    let db_url = std::env::var("OS_V3_DATABASE_URL").unwrap_or_else(|_| {
        "postgresql://imaginos:imaginos_secure_2026@127.0.0.1:5432/os_core_db".to_string()
    });

    // Use psql directly — no SeaORM dep needed in watchdog
    let output = std::process::Command::new("psql")
        .arg(&db_url)
        .arg("-c")
        .arg(
            "DELETE FROM audit_log WHERE \
             (retention_until IS NOT NULL AND retention_until < NOW()) \
             OR (capability_used IN ('ingress_trace', 'sentinel_edge') \
                 AND timestamp < NOW() - INTERVAL '7 days') \
             OR timestamp < NOW() - INTERVAL '30 days'; \
             VACUUM (ANALYZE) audit_log;",
        )
        .env("PGPASSWORD", std::env::var("PGPASSWORD").unwrap_or_default())
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let out = String::from_utf8_lossy(&o.stdout);
            let deleted: u64 = out
                .lines()
                .find(|l| l.starts_with("DELETE"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            if deleted > 0 {
                info!("🧹 [Watchdog] Purged {} expired audit_log rows", deleted);
            }
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            if !err.is_empty() {
                warn!("🐕 [Watchdog] audit_log purge warning: {}", err.trim());
            }
        }
        Err(e) => {
            warn!("🐕 [Watchdog] audit_log purge skipped (psql unavailable): {}", e);
        }
    }
}

/// Log a watchdog event to the persistent log file
fn log_watchdog_event(emergency: &Emergency) {
    let now = {
        let d = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        format!("{}", d.as_secs())
    };
    let severity_str = match emergency.severity {
        Severity::AutoHealed => "auto_healed",
        Severity::Warning => "warning",
        Severity::Critical => "critical",
    };

    let entry = serde_json::json!({
        "timestamp": now,
        "severity": severity_str,
        "category": emergency.category,
        "service": emergency.service,
        "message": emergency.message,
        "action_taken": emergency.action_taken,
    });

    if let Ok(line) = serde_json::to_string(&entry) {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(WATCHDOG_LOG).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(WATCHDOG_LOG)
            .ok();
        if let Some(ref mut f) = file {
            use std::io::{Seek, SeekFrom, Write};
            let _ = writeln!(f, "{}", line);
            // Rotate when file exceeds 2 MB — keep last 2000 lines
            if f.seek(SeekFrom::End(0)).unwrap_or(0) > 2 * 1024 * 1024 {
                drop(file);
                let _ = rotate_watchdog_log(2000);
            }
        }
    }
}

fn rotate_watchdog_log(keep_lines: usize) {
    if let Ok(content) = std::fs::read_to_string(WATCHDOG_LOG) {
        let lines: Vec<&str> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        if lines.len() > keep_lines {
            let tail = lines[lines.len() - keep_lines..].join("\n");
            let _ = std::fs::write(WATCHDOG_LOG, format!("{tail}\n"));
        }
    }
}

/// Write a critical alert to the alert file for external consumers
/// (Telegram bots, WhatsApp gateways, Imaginclaw UI can poll this)
fn write_alert(emergency: &Emergency) {
    let now = {
        let d = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        format!("{}", d.as_secs())
    };
    let alert = serde_json::json!({
        "timestamp": now,
        "severity": "critical",
        "category": emergency.category,
        "service": emergency.service,
        "message": emergency.message,
        "action_taken": emergency.action_taken,
        "acknowledged": false,
    });

    if let Ok(line) = serde_json::to_string(&alert) {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(ALERT_FILE)
            .ok();
        if let Some(ref mut f) = file {
            use std::io::Write;
            let _ = writeln!(f, "{}", line);
        }
    }
}
