//! Docker Container Abstraction Layer
//!
//! Replaces legacy `smartos-mcp` executing `docker ps --format "{{...}}"`.
//! Maintains an internal thread-safe cache to avoid saturating the local Docker socket.

use lazy_static::lazy_static;
use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::{Duration, Instant};
use tracing::{debug, error};

const CACHE_TTL: Duration = Duration::from_secs(30);

lazy_static! {
    static ref SERVICE_CACHE: RwLock<ServiceCache> = RwLock::new(ServiceCache {
        services: Vec::new(),
        last_updated: Instant::now() - Duration::from_secs(86400), // Force initial fetch
    });

    // Legacy parsing: 0.0.0.0:8000->8000/tcp
    static ref PORT_MAPPING_REGEX: Regex = Regex::new(r"(\d+\.\d+\.\d+\.\d+)?:?(\d+)->(\d+)").unwrap();
}

struct ServiceCache {
    services: Vec<DockerService>,
    last_updated: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub host: String,
    pub container: String,
    pub bind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerService {
    pub name: String,
    pub image: String,
    pub ports: Vec<PortMapping>,
    pub status: String, // "healthy" | "unhealthy"
    pub uptime: String,
    pub category: String, // "gpu-image" | "gpu-llm" | "utility" | "memory" etc.
}

/// Discovers running containers matching the Hera3 context bounds.
/// Caches the output to prevent Docker Daemon saturation (Subagent models run very fast).
pub fn discover_docker_services() -> Vec<DockerService> {
    // 1. Check TTL cache
    {
        let cache = SERVICE_CACHE.read();
        if cache.last_updated.elapsed() < CACHE_TTL {
            return cache.services.clone();
        }
    } // Drop read lock

    // 2. Poll Docker Engine
    let output = match Command::new("docker")
        .args([
            "ps",
            "--format",
            "{{.Names}}|{{.Image}}|{{.Ports}}|{{.Status}}",
        ])
        .output()
    {
        Ok(o) => {
            if !o.status.success() {
                error!("Docker daemon query failed. Access context restriction?");
                return Vec::new();
            }
            String::from_utf8_lossy(&o.stdout).to_string()
        }
        Err(e) => {
            error!("Failed to execute 'docker ps': {}", e);
            return Vec::new(); // Fallback empty
        }
    };

    // 3. Parse output iteratively
    let mut services = Vec::new();
    for line in output.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 4 {
            continue;
        }

        let name = parts[0];
        let image = parts[1].split(':').next().unwrap_or(parts[1]).to_string(); // Strip tag
        let ports_raw = parts[2];
        let status_raw = parts[3];

        let mut ports = Vec::new();
        for pm in ports_raw.split(',') {
            let pm = pm.trim();
            if pm.is_empty() {
                continue;
            }
            if let Some(caps) = PORT_MAPPING_REGEX.captures(pm) {
                ports.push(PortMapping {
                    bind: caps
                        .get(1)
                        .map(|m: regex::Match| m.as_str())
                        .unwrap_or("0.0.0.0")
                        .to_string(),
                    host: caps.get(2).unwrap().as_str().to_string(),
                    container: caps.get(3).unwrap().as_str().to_string(),
                });
            }
        }

        // Categorize based on strict namespace boundaries (Legacy index.js rules)
        let category = if name.contains("draw") {
            "gpu-image"
        } else if name.contains("ai") || name.contains("llama") {
            "gpu-llm"
        } else if name.contains("comfy") {
            "gpu-video"
        } else if name.contains("hear") || name.contains("whisper") {
            "gpu-audio"
        } else if name.contains("mongo") || name.contains("postgres") {
            "database"
        } else if name.contains("minio") || name.contains("n8n") {
            "infrastructure"
        } else if name.contains("explore") || name.contains("write") || name.contains("digest") {
            "utility"
        } else if name.contains("remember") || name.contains("qdrant") {
            "memory"
        } else {
            "other"
        };

        let is_healthy = status_raw.contains("Up");

        services.push(DockerService {
            name: name.to_string(),
            image,
            ports,
            status: if is_healthy {
                "healthy".to_string()
            } else {
                "unhealthy".to_string()
            },
            uptime: status_raw.to_string(),
            category: category.to_string(),
        });
    }

    debug!(
        "Pivoting cache: {} active Docker services discovered",
        services.len()
    );

    // 4. Update TTL Cache globally
    let mut cache = SERVICE_CACHE.write();
    cache.services = services.clone();
    cache.last_updated = Instant::now();

    services
}

#[cfg(test)]
mod tests {
    use super::*;

    // Not mocking standard library external dependencies in simple unit test,
    // but demonstrating compiling integrity bounds.
}
