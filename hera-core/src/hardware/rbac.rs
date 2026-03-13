//! Quality Tiering and Hardware Fallback Mechanism (RBAC Logic)
//!
//! Maps an authenticated session context to physical GPU resources.
//! It handles the legacy overrides (quality, turbo, instant) to map to
//! native SwarmUI execution configurations and engine target ports.

use crate::hardware::docker::discover_docker_services;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Identifies the capability surface assigned to the hardware.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuTier {
    pub tier_name: String,
    pub port: u16,
    pub steps: u32,
    pub description: String,
    pub fallback: bool,
    pub role: String, // Attached back to the user object dynamically
}

impl GpuTier {
    pub fn admin() -> Self {
        Self {
            tier_name: "quality".to_string(),
            port: 7800,
            steps: 25,
            description: "High quality generation - admin tier".to_string(),
            fallback: false,
            role: "admin".to_string(),
        }
    }

    pub fn premium() -> Self {
        Self {
            tier_name: "turbo".to_string(),
            port: 7801,
            steps: 12,
            description: "Fast high-res generation - premium tier".to_string(),
            fallback: false,
            role: "premium".to_string(),
        }
    }

    pub fn guest() -> Self {
        Self {
            tier_name: "instant".to_string(),
            port: 7802,
            steps: 6,
            description: "Instant low-res previews - guest tier".to_string(),
            fallback: false,
            role: "guest".to_string(),
        }
    }
}

/// Takes a simulated or resolved user role ("admin" | "premium" | "guest")
/// and determines the optimal GPU pipeline to assign.
/// Checks Docker cache natively to automatically reroute if target GPU container is down.
pub fn get_gpu_endpoint(user_role: &str, engine_override: Option<&str>) -> GpuTier {
    let role = user_role.to_string();

    // 1. Hard Overrides (System bypassing RBAC)
    let mut initial_tier = match engine_override {
        Some("quality") => {
            let mut t = GpuTier::admin();
            t.role = role.clone();
            t
        }
        Some("turbo") => {
            let mut t = GpuTier::premium();
            t.role = role.clone();
            t
        }
        Some("instant") => {
            let mut t = GpuTier::guest();
            t.role = role.clone();
            t
        }
        _ => match role.as_str() {
            "admin" => GpuTier::admin(),
            "premium" => GpuTier::premium(),
            _ => GpuTier::guest(), // Default surface
        },
    };

    initial_tier.role = role.clone();

    // 2. Fallback Active Check
    // If Admin needs 7800, but container is offline, fallback to 7802 or 7801
    let services = discover_docker_services();
    let draw_services: Vec<_> = services
        .iter()
        .filter(|s| s.category == "gpu-image")
        .collect();

    let target_port_str = initial_tier.port.to_string();
    let is_running = draw_services.iter().any(|svc| {
        svc.ports
            .iter()
            .any(|mapping| mapping.host == target_port_str)
    });

    if !is_running && !draw_services.is_empty() {
        // Degrade to the first active port
        if let Some(fallback_svc) = draw_services.first() {
            if let Some(pm) = fallback_svc.ports.first() {
                if let Ok(p) = pm.host.parse::<u16>() {
                    warn!(
                        "GPU Fallback triggered: Port {} unavailable, redirecting to {}",
                        initial_tier.port, p
                    );
                    initial_tier.port = p;
                    initial_tier.fallback = true;
                }
            }
        }
    }

    initial_tier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_mapping() {
        assert_eq!(GpuTier::admin().steps, 25);
        assert_eq!(GpuTier::admin().port, 7800);
        assert_eq!(GpuTier::guest().steps, 6);
    }
}
