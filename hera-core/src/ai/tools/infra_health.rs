//! Infrastructure health tool executors: diagnose, system_status, service_restart, pm2_logs
use crate::ai::tool_executor::{
    canonical_app_search_terms, load_canonical_app_registry, text_contains_app_alias, ToolCall,
    ToolResult,
};
use serde_json::Value;
use std::process::Command;
use tracing::info;

pub(crate) async fn execute_diagnose_services(call: &ToolCall) -> ToolResult {
    let service_filter = call
        .arguments
        .get("service_filter")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_lowercase();
    let include_logs = call
        .arguments
        .get("include_logs")
        .and_then(|b| b.as_bool())
        .unwrap_or(true);
    let service_terms = if service_filter.is_empty() {
        Vec::new()
    } else {
        canonical_app_search_terms(&service_filter)
    };

    let mut report = String::new();
    report.push_str("🏥 ImagineOS Service Diagnostic Report\n");
    report.push_str("═══════════════════════════════════════\n\n");

    // ── 1. Parse services.conf to get expected service→port map ──
    let services_conf_path = "/home/paulo/Programs/apps/OS/etc/sentinel/services.conf";
    let mut expected_services: Vec<(String, u16)> = Vec::new();

    if let Ok(content) = std::fs::read_to_string(services_conf_path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            // Format: hostname = port [options]
            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() == 2 {
                let host = parts[0].trim().to_string();
                let port_str = parts[1].split_whitespace().next().unwrap_or("0");
                if let Ok(port) = port_str.parse::<u16>() {
                    expected_services.push((host, port));
                }
            }
        }
    } else {
        report.push_str("⚠️ Could not read services.conf — skipping expected-service analysis\n");
    }

    // Deduplicate ports (multiple hostnames can point to same port)
    let mut unique_ports: std::collections::HashMap<u16, Vec<String>> =
        std::collections::HashMap::new();
    for (host, port) in &expected_services {
        unique_ports.entry(*port).or_default().push(host.clone());
    }

    // ── 2. Get PM2 process list ──
    let mut pm2_services: Vec<(String, String, u64, u64)> = Vec::new(); // (name, status, restarts, pid)
    if let Ok(output) = std::process::Command::new("pm2").arg("jlist").output()
        && output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(procs) = serde_json::from_str::<Vec<serde_json::Value>>(&out_str) {
                for proc in &procs {
                    let name = proc
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let status = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("status"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let restarts = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("restart_time"))
                        .and_then(|r| r.as_u64())
                        .unwrap_or(0);
                    let pid = proc.get("pid").and_then(|p| p.as_u64()).unwrap_or(0);
                    pm2_services.push((name, status, restarts, pid));
                }
            }
        }

    // ── 3. Get actual port listeners via ss ──
    let mut port_owners: std::collections::HashMap<u16, String> = std::collections::HashMap::new();
    if let Ok(output) = std::process::Command::new("ss").args(["-tlnp"]).output()
        && output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines().skip(1) {
                // Parse: LISTEN  0  4096  0.0.0.0:5150  0.0.0.0:*  users:(("proc",pid=X,fd=Y))
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let addr = parts[3];
                    if let Some(port_str) = addr.rsplit(':').next()
                        && let Ok(port) = port_str.parse::<u16>() {
                            // Extract process name from users:((...)) field
                            let proc_info = parts.get(5).unwrap_or(&"");
                            let proc_name = if let Some(start) = proc_info.find("((\"") {
                                let after = &proc_info[start + 3..];
                                after.split('"').next().unwrap_or("unknown").to_string()
                            } else {
                                "unknown".to_string()
                            };
                            port_owners.insert(port, proc_name);
                        }
                }
            }
        }

    // ── 4. HTTP-probe each unique port ──
    let mut port_status: std::collections::HashMap<u16, (u16, String)> =
        std::collections::HashMap::new(); // port -> (http_code, error)
    for &port in unique_ports.keys() {
        if !service_terms.is_empty() {
            // Check if any hostname for this port matches the filter
            let hosts = unique_ports.get(&port).cloned().unwrap_or_default();
            if !hosts
                .iter()
                .any(|h| text_contains_app_alias(h, &service_terms))
            {
                continue;
            }
        }

        let url = format!("http://127.0.0.1:{}/", port);
        match std::process::Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "--connect-timeout",
                "2",
                &url,
            ])
            .output()
        {
            Ok(output) => {
                let code_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let code: u16 = code_str.parse().unwrap_or(0);
                if code == 0 {
                    port_status.insert(port, (0, "Connection refused / timeout".to_string()));
                } else {
                    port_status.insert(port, (code, String::new()));
                }
            }
            Err(e) => {
                port_status.insert(port, (0, format!("curl failed: {}", e)));
            }
        }
    }

    // ── 5. Correlate and produce report ──
    let mut healthy: Vec<String> = Vec::new();
    let mut degraded: Vec<String> = Vec::new();
    let mut down: Vec<String> = Vec::new();
    let mut root_causes: Vec<String> = Vec::new();
    let mut proposed_fixes: Vec<String> = Vec::new();

    // Sort ports for consistent output
    let mut sorted_ports: Vec<u16> = unique_ports.keys().cloned().collect();
    sorted_ports.sort();

    for port in &sorted_ports {
        let hosts = unique_ports.get(port).cloned().unwrap_or_default();
        let host_label = hosts
            .first()
            .cloned()
            .unwrap_or_else(|| format!("port:{}", port));

        if !service_terms.is_empty()
            && !hosts
                .iter()
                .any(|h| text_contains_app_alias(h, &service_terms))
            {
                continue;
            }

        let port_owner = port_owners.get(port);
        let http = port_status.get(port);

        match (port_owner, http) {
            // Port is listening AND responds with 2xx/3xx
            (Some(owner), Some((code, _))) if *code >= 200 && *code < 400 => {
                healthy.push(format!(
                    "✅ {} (:{}) → HTTP {} [process: {}]",
                    host_label, port, code, owner
                ));
            }
            // Port is listening but responds with 4xx/5xx
            (Some(owner), Some((code, _))) if *code >= 400 => {
                degraded.push(format!(
                    "⚠️ {} (:{}) → HTTP {} [process: {}]",
                    host_label, port, code, owner
                ));
                if *code == 500 {
                    root_causes.push(format!("Port {} ({}) returns 500 — likely an unhandled exception or template rendering error in {}", port, host_label, owner));
                    proposed_fixes.push(format!(
                        "Check error logs: `pm2 logs {} --err --lines 20`",
                        owner.replace("_rust-cl", "-rust").replace("-cli", "")
                    ));
                }
            }
            // Port is NOT listening at all
            (None, _) => {
                down.push(format!("🔴 {} (:{}) → NO LISTENER", host_label, port));
                // Check if there's a PM2 process that should own this port
                let possible_pm2 = pm2_services.iter().find(|(name, _, _, _)| {
                    let host_base = host_label.split('.').next().unwrap_or("").to_lowercase();
                    let pm2_aliases = canonical_app_search_terms(name);
                    text_contains_app_alias(&host_label, &pm2_aliases)
                        || pm2_aliases
                            .iter()
                            .any(|alias: &String| host_base.contains(alias) || alias.contains(&host_base))
                });
                if let Some((pm2_name, pm2_status, restarts, _)) = possible_pm2 {
                    root_causes.push(format!(
                        "Port {} ({}) has no listener but PM2 shows '{}' as {} with {} restarts — process may have crashed or port is misconfigured",
                        port, host_label, pm2_name, pm2_status, restarts
                    ));
                    proposed_fixes.push(format!("Try: `pm2 restart {}`", pm2_name));
                } else {
                    root_causes.push(format!(
                        "Port {} ({}) has no listener and NO matching PM2 process — service may not be registered in PM2",
                        port, host_label
                    ));
                    proposed_fixes.push("Register the service in PM2 or verify the port in services.conf".to_string());
                }
            }
            // Port listening but HTTP probe returned 0 (connection issues)
            (Some(owner), Some((0, err))) => {
                degraded.push(format!(
                    "⚠️ {} (:{}) → Connection issue: {} [process: {}]",
                    host_label, port, err, owner
                ));
            }
            _ => {
                degraded.push(format!("⚠️ {} (:{}) → Unknown state", host_label, port));
            }
        }
    }

    // Check for port conflicts (two different expected services on the same port)
    for (port, hosts) in &unique_ports {
        if let Some(owner) = port_owners.get(port) {
            // Check if the owner process name matches what we'd expect
            let owner_aliases = canonical_app_search_terms(owner);
            let expected_any = hosts
                .iter()
                .any(|h| text_contains_app_alias(h, &owner_aliases));
            if !expected_any && !owner.contains("sentinel") {
                root_causes.push(format!(
                    "🔀 PORT CONFLICT: Port {} is expected for {:?} but is owned by process '{}'",
                    port, hosts, owner
                ));
                proposed_fixes.push(format!(
                    "Check if '{}' should be on port {} or if there's a port collision. Verify config files.",
                    owner, port
                ));
            }
        }
    }

    // Check for PM2 crash loops
    for (name, status, restarts, _) in &pm2_services {
        if *restarts > 10 {
            root_causes.push(format!(
                "🔄 CRASH LOOP: PM2 service '{}' has {} restarts (status: {}) — likely a persistent error preventing stable startup",
                name, restarts, status
            ));
            proposed_fixes.push(format!("Investigate root cause: `pm2 logs {} --err --lines 30` then fix the underlying error before restarting", name));
        }
    }

    // Check VRAM exhaustion
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        && output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for (i, line) in out_str.lines().enumerate() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 2 {
                    let used: f64 = parts[0].trim().parse().unwrap_or(0.0);
                    let total: f64 = parts[1].trim().parse().unwrap_or(1.0);
                    let pct = (used / total) * 100.0;
                    if pct > 95.0 {
                        root_causes.push(format!(
                            "🔥 VRAM EXHAUSTION: GPU {} is at {:.0}% VRAM ({:.0}MB / {:.0}MB) — new GPU-dependent services will fail to start",
                            i, pct, used, total
                        ));
                        proposed_fixes.push(format!("Free GPU {} VRAM by stopping unused GPU processes: `nvidia-smi` then kill the heaviest one", i));
                    }
                }
            }
        }

    // Check disk space
    if let Ok(output) = std::process::Command::new("df")
        .args(["-h", "--output=target,pcent,avail", "/", "/home"])
        .output()
        && output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let mount = parts[0];
                    let pct_str = parts[1].trim_end_matches('%');
                    if let Ok(pct) = pct_str.parse::<u32>()
                        && pct > 90 {
                            root_causes.push(format!(
                                "💾 DISK FULL: {} is at {}% usage (only {} free) — services will crash on write",
                                mount, pct, parts[2]
                            ));
                            proposed_fixes.push(format!(
                                "Free disk space on {}: check /tmp, Docker images, PM2 logs, and build artifacts",
                                mount
                            ));
                        }
                }
            }
        }

    // Check WireGuard tunnel status
    if let Ok(output) = std::process::Command::new("wg").arg("show").output()
        && output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if out_str.trim().is_empty() {
                root_causes.push("🔒 WIREGUARD DOWN: No active WireGuard tunnels — external traffic cannot reach services".to_string());
                proposed_fixes.push("Bring up WireGuard: `sudo wg-quick up wg0`".to_string());
            } else {
                // Check for handshake staleness (last handshake > 5 minutes ago)
                for line in out_str.lines() {
                    if line.contains("latest handshake:")
                        && (line.contains("minutes") || line.contains("hours"))
                        && line.contains("hour") {
                            root_causes.push(format!(
                                "🔒 WIREGUARD STALE: Tunnel peer handshake is stale ({})",
                                line.trim()
                            ));
                            proposed_fixes.push("Check WireGuard peer connectivity: `sudo wg-quick down wg0 && sudo wg-quick up wg0`".to_string());
                        }
                }
            }
        }

    // Check Caddy/Sentinel reverse proxy (port 3000)
    if let Some(&sentinel_port) = sorted_ports.iter().find(|p| **p == 3000)
        && port_owners.get(&sentinel_port).is_none() {
            root_causes.push("🚪 SENTINEL DOWN: Port 3000 (Caddy/Sentinel reverse proxy) has no listener — ALL external traffic is blocked".to_string());
            proposed_fixes
                .push("CRITICAL: Restart Sentinel immediately: `pm2 restart sentinel`".to_string());
        }

    // ── 6. Format the final report ──
    if !healthy.is_empty() {
        report.push_str(&format!("HEALTHY ({}):\n", healthy.len()));
        for s in &healthy {
            report.push_str(&format!("  {}\n", s));
        }
        report.push('\n');
    }
    if !degraded.is_empty() {
        report.push_str(&format!("DEGRADED ({}):\n", degraded.len()));
        for s in &degraded {
            report.push_str(&format!("  {}\n", s));
        }
        report.push('\n');
    }
    if !down.is_empty() {
        report.push_str(&format!("DOWN ({}):\n", down.len()));
        for s in &down {
            report.push_str(&format!("  {}\n", s));
        }
        report.push('\n');
    }

    if !root_causes.is_empty() {
        report.push_str("ROOT CAUSE HYPOTHESES:\n");
        for (i, rc) in root_causes.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, rc));
        }
        report.push('\n');
    }

    if !proposed_fixes.is_empty() {
        report.push_str("PROPOSED FIXES:\n");
        for (i, fix) in proposed_fixes.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, fix));
        }
        report.push('\n');
    }

    // ── 7. Include recent error logs if requested ──
    if include_logs && !degraded.is_empty() {
        report.push_str("RECENT ERROR LOGS (degraded services):\n");
        for entry in &degraded {
            // Extract PM2 process name from the entry
            if let Some(proc_start) = entry.find("process: ") {
                let proc_name = &entry[proc_start + 9..];
                let proc_name = proc_name.trim_end_matches(']');
                // Normalize process name for PM2 log path
                let pm2_name = proc_name
                    .replace("_rust-cl", "-rust")
                    .replace("-cli", "")
                    .replace("_", "-");
                let log_path = format!("/home/paulo/.pm2/logs/{}-error.log", pm2_name);
                if let Ok(content) = std::fs::read_to_string(&log_path) {
                    let lines: Vec<&str> = content.lines().collect();
                    let start = if lines.len() > 10 {
                        lines.len() - 10
                    } else {
                        0
                    };
                    report.push_str(&format!("  ── {} ──\n", pm2_name));
                    for line in &lines[start..] {
                        report.push_str(&format!("    {}\n", line));
                    }
                }
            }
        }
    }

    let total = healthy.len() + degraded.len() + down.len();
    let summary = format!(
        "SUMMARY: {} services checked — {} healthy, {} degraded, {} down",
        total,
        healthy.len(),
        degraded.len(),
        down.len()
    );
    report.push_str(&format!("\n{}\n", summary));

    info!("🏥 [Hera] Service diagnostic complete: {}", summary);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: report,
    }
}

pub(crate) async fn execute_system_status(call: &ToolCall) -> ToolResult {
    let mut report = String::new();

    // 1. RAM from /proc/meminfo
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        let mut total = 0.0_f64;
        let mut available = 0.0_f64;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse::<f64>()
                    .unwrap_or(0.0)
                    / 1024.0
                    / 1024.0;
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse::<f64>()
                    .unwrap_or(0.0)
                    / 1024.0
                    / 1024.0;
            }
        }
        let used = total - available;
        report.push_str(&format!(
            "RAM: {:.1}GB used / {:.1}GB total ({:.1}GB free)\n",
            used, total, available
        ));
    }

    // 2. CPU Load from /proc/loadavg
    if let Ok(loadavg) = std::fs::read_to_string("/proc/loadavg") {
        let parts: Vec<&str> = loadavg.split_whitespace().collect();
        if parts.len() >= 3 {
            report.push_str(&format!(
                "CPU Load Average: {} (1m) {} (5m) {} (15m)\n",
                parts[0], parts[1], parts[2]
            ));
        }
    }

    // 3. Uptime
    if let Ok(output) = std::process::Command::new("uptime").arg("-p").output() {
        let uptime = String::from_utf8_lossy(&output.stdout).trim().to_string();
        report.push_str(&format!("Uptime: {}\n", uptime));
    }

    // 4. GPU status via nvidia-smi
    match std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,memory.free")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => {
            let out_str = String::from_utf8_lossy(&output.stdout);
            report.push_str("\nGPU Status:\n");
            for line in out_str.lines() {
                let parts: Vec<&str> = line.split(", ").collect();
                if parts.len() == 7 {
                    report.push_str(&format!(
                        "  GPU {}: {} | Temp: {}°C | Load: {}% | VRAM: {}MB / {}MB ({}MB free)\n",
                        parts[0].trim(), parts[1].trim(), parts[2].trim(),
                        parts[3].trim(), parts[4].trim(), parts[5].trim(), parts[6].trim()
                    ));
                }
            }
        }
        _ => {
            report.push_str("\nGPU: nvidia-smi not available or failed.\n");
        }
    }

    // 5. GPU process list
    match std::process::Command::new("nvidia-smi")
        .arg("--query-compute-apps=pid,name,used_memory")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if !out_str.trim().is_empty() {
                report.push_str("\nGPU Processes:\n");
                for line in out_str.lines() {
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() == 3 {
                        let proc_name =
                            parts[1].trim().split('/').next_back().unwrap_or(parts[1].trim());
                        report.push_str(&format!(
                            "  PID {} | {} | {}MB VRAM\n",
                            parts[0].trim(),
                            proc_name,
                            parts[2].trim()
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    // 6. PM2 services status
    // Pre-load port listeners to map PID to Ports
    let mut port_by_pid: std::collections::HashMap<u64, Vec<u16>> =
        std::collections::HashMap::new();
    if let Ok(output) = std::process::Command::new("ss").args(["-tlnp"]).output()
        && output.status.success() {
            let out_str = String::from_utf8_lossy(&output.stdout);
            for line in out_str.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5
                    && let Some(port_str) = parts[3].rsplit(':').next()
                        && let Ok(port) = port_str.parse::<u16>() {
                            let proc_info = parts.get(5).unwrap_or(&"");
                            if let Some(start) = proc_info.find("pid=") {
                                let after = &proc_info[start + 4..];
                                let pid_str = after.split(',').next().unwrap_or("0");
                                if let Ok(pid) = pid_str.parse::<u64>() {
                                    port_by_pid.entry(pid).or_default().push(port);
                                }
                            }
                        }
            }
        }

    match std::process::Command::new("pm2").arg("jlist").output() {
        Ok(output) if output.status.success() => {
            let out_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(procs) = serde_json::from_str::<Vec<serde_json::Value>>(&out_str) {
                report.push_str(&format!("\nPM2 Services ({} total):\n", procs.len()));
                for proc in &procs {
                    let name = proc.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    let status = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("status"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("?");
                    let restarts = proc
                        .get("pm2_env")
                        .and_then(|e| e.get("restart_time"))
                        .and_then(|r| r.as_u64())
                        .unwrap_or(0);
                    let pid = proc.get("pid").and_then(|p| p.as_u64()).unwrap_or(0);

                    let emoji = if status == "online" { "🟢" } else { "🔴" };
                    let crash_flag = if restarts > 10 {
                        " ⚠️ CRASH LOOP"
                    } else {
                        ""
                    };

                    let ports = port_by_pid.get(&pid);
                    let port_info = if let Some(p) = ports {
                        format!(" (ports: {:?})", p)
                    } else if status == "online"
                        && !name.contains("argus")
                        && !name.contains("imagin")
                        && !name.contains("memento")
                    {
                        " (no listener)".to_string()
                    } else {
                        "".to_string()
                    };

                    report.push_str(&format!(
                        "  {} {} [{}] restarts: {}{}{}\n",
                        emoji, name, status, restarts, port_info, crash_flag
                    ));
                }
            }
        }
        _ => {
            report.push_str("\nPM2: Not available\n");
        }
    }

    info!("🖥️ [Hera] System status report generated");
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: report,
    }
}

/// Auto-heal: restart a PM2 service by name.
/// Ava can now fix problems, not just report them.
pub(crate) async fn execute_service_restart(call: &ToolCall) -> ToolResult {
    let service_name = call
        .arguments
        .get("service_name")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let flush_logs = call
        .arguments
        .get("flush_logs")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);

    if service_name.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'service_name' parameter. Provide the PM2 process name (e.g., 'vetra-rust').".into(),
        };
    }

    // Safety: sanitize service name to prevent injection
    let sanitized: String = service_name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_' || *ch == '.')
        .collect();

    let mut report = String::new();

    // Step 1: Capture pre-restart state
    let pre_status = std::process::Command::new("pm2")
        .args(["describe", &sanitized])
        .output();
    if let Ok(output) = &pre_status {
        let out_str = String::from_utf8_lossy(&output.stdout);
        if out_str.contains("doesn't exist") || out_str.contains("Process not found") {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!(
                    "PM2 process '{}' not found. Run `pm2 list` to see available services.",
                    sanitized
                ),
            };
        }
    }

    // Step 2: Optionally flush logs before restart
    if flush_logs {
        let _ = std::process::Command::new("pm2")
            .args(["flush", &sanitized])
            .output();
        report.push_str(&format!("🗑️ Flushed logs for '{}'\n", sanitized));
    }

    // Step 3: Read last 5 error lines before restart (for context)
    let pm2_home = std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string());
    let err_log_path = format!("{}/.pm2/logs/{}-error.log", pm2_home, sanitized);
    if let Ok(content) = std::fs::read_to_string(&err_log_path) {
        let lines: Vec<&str> = content.lines().collect();
        let start = if lines.len() > 5 { lines.len() - 5 } else { 0 };
        if !lines[start..].is_empty() {
            report.push_str("Last errors before restart:\n");
            for line in &lines[start..] {
                report.push_str(&format!("  {}", line));
                report.push('\n');
            }
        }
    }

    // Step 4: Execute restart
    match std::process::Command::new("pm2")
        .args(["restart", &sanitized])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                // Step 5: Wait a moment, then verify the service came back
                std::thread::sleep(std::time::Duration::from_secs(2));

                let is_online = if let Ok(verify) =
                    std::process::Command::new("pm2").args(["jlist"]).output()
                {
                    let out_str = String::from_utf8_lossy(&verify.stdout);
                    if let Ok(procs) = serde_json::from_str::<Vec<serde_json::Value>>(&out_str) {
                        procs.iter().any(|proc| {
                            let name = proc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                            let status = proc
                                .get("pm2_env")
                                .and_then(|e| e.get("status"))
                                .and_then(|s| s.as_str())
                                .unwrap_or("");
                            name == sanitized && status == "online"
                        })
                    } else {
                        false
                    }
                } else {
                    false
                };

                if is_online {
                    report.push_str(&format!(
                        "\n✅ Service '{}' restarted successfully and is ONLINE.",
                        sanitized
                    ));
                    info!(
                        "🔧 [Hera] Auto-heal: '{}' restarted successfully",
                        sanitized
                    );
                } else {
                    report.push_str(&format!("\n⚠️ Service '{}' was restarted but is NOT online yet. It may need more time or has a startup error.", sanitized));
                    report.push_str(
                        "\nRecommendation: Use read_pm2_logs to check for startup errors.",
                    );
                }

                ToolResult {
                    name: call.name.clone(),
                    success: is_online,
                    output: report,
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("PM2 restart failed for '{}': {}", sanitized, stderr),
                }
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to execute pm2 restart: {}", e),
        },
    }
}

/// Read PM2 logs for a specific service.
/// Gives Ava deep per-service log access beyond the centralized JSONL file.
pub(crate) async fn execute_read_pm2_logs(call: &ToolCall) -> ToolResult {
    let service_name = call
        .arguments
        .get("service_name")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let log_type = call
        .arguments
        .get("log_type")
        .and_then(|t| t.as_str())
        .unwrap_or("error");
    let lines = call
        .arguments
        .get("lines")
        .and_then(|l| l.as_i64())
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let search = call
        .arguments
        .get("search")
        .and_then(|s| s.as_str())
        .unwrap_or("");

    if service_name.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing 'service_name' parameter.".into(),
        };
    }

    let sanitized: String = service_name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_' || *ch == '.')
        .collect();

    let pm2_home = std::env::var("HOME").unwrap_or_else(|_| "/home/paulo".to_string());

    let read_log_file = |suffix: &str| -> String {
        let path = format!("{}/.pm2/logs/{}-{}.log", pm2_home, sanitized, suffix);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                let all_lines: Vec<&str> = content.lines().collect();
                let filtered: Vec<&&str> = if search.is_empty() {
                    all_lines.iter().collect()
                } else {
                    let search_lower = search.to_lowercase();
                    all_lines
                        .iter()
                        .filter(|l| l.to_lowercase().contains(&search_lower))
                        .collect()
                };
                let start = if filtered.len() > lines {
                    filtered.len() - lines
                } else {
                    0
                };
                filtered[start..]
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            Err(_) => format!(
                "(no {} log file found at {}/.pm2/logs/{}-{}.log)",
                suffix, pm2_home, sanitized, suffix
            ),
        }
    };

    let mut result = String::new();
    match log_type {
        "output" => {
            result.push_str(&format!(
                "=== PM2 OUTPUT LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("out"));
        }
        "both" => {
            result.push_str(&format!(
                "=== PM2 ERROR LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("error"));
            result.push_str(&format!(
                "\n\n=== PM2 OUTPUT LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("out"));
        }
        _ => {
            result.push_str(&format!(
                "=== PM2 ERROR LOG for '{}' (last {} lines) ===\n",
                sanitized, lines
            ));
            result.push_str(&read_log_file("error"));
        }
    }

    info!("📋 [Hera] Read PM2 {} logs for '{}'", log_type, sanitized);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: result,
    }
}

