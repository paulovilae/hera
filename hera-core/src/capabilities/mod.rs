use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityId {
    LocalLlm,
    Vision,
    AudioTts,
    AudioStt,
    DesktopControl,
    ExecutionTools,
    McpBridge,
}

impl CapabilityId {
    pub fn as_str(self) -> &'static str {
        match self {
            CapabilityId::LocalLlm => "local_llm",
            CapabilityId::Vision => "vision",
            CapabilityId::AudioTts => "audio_tts",
            CapabilityId::AudioStt => "audio_stt",
            CapabilityId::DesktopControl => "desktop_control",
            CapabilityId::ExecutionTools => "execution_tools",
            CapabilityId::McpBridge => "mcp_bridge",
        }
    }

    pub fn startup_cost(self) -> &'static str {
        match self {
            CapabilityId::LocalLlm => "high",
            CapabilityId::Vision => "medium",
            CapabilityId::AudioTts => "medium",
            CapabilityId::AudioStt => "medium",
            CapabilityId::DesktopControl => "low",
            CapabilityId::ExecutionTools => "medium",
            CapabilityId::McpBridge => "low",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CapabilityStatus {
    pub id: CapabilityId,
    pub compiled: bool,
    pub env_enabled: bool,
    pub startup_cost: &'static str,
    pub note: &'static str,
}

#[derive(Debug, Clone)]
pub struct CapabilityRegistry {
    statuses: Vec<CapabilityStatus>,
}

impl CapabilityRegistry {
    pub fn detect() -> Self {
        Self {
            statuses: vec![
                CapabilityStatus {
                    id: CapabilityId::LocalLlm,
                    compiled: cfg!(feature = "local-llm"),
                    env_enabled: env_flag("HERA_ENABLE_LLM", true),
                    startup_cost: CapabilityId::LocalLlm.startup_cost(),
                    note: "Local-first text generation and routing",
                },
                CapabilityStatus {
                    id: CapabilityId::Vision,
                    compiled: cfg!(feature = "vision"),
                    env_enabled: true,
                    startup_cost: CapabilityId::Vision.startup_cost(),
                    note: "Vision requests routed through the local-compatible engine",
                },
                CapabilityStatus {
                    id: CapabilityId::AudioTts,
                    compiled: cfg!(feature = "audio"),
                    env_enabled: env_flag("HERA_ENABLE_PARLER", true),
                    startup_cost: CapabilityId::AudioTts.startup_cost(),
                    note: "Parler TTS voice synthesis",
                },
                CapabilityStatus {
                    id: CapabilityId::AudioStt,
                    compiled: cfg!(feature = "audio"),
                    env_enabled: env_flag("HERA_ENABLE_WHISPER", true),
                    startup_cost: CapabilityId::AudioStt.startup_cost(),
                    note: "Whisper speech transcription",
                },
                CapabilityStatus {
                    id: CapabilityId::DesktopControl,
                    compiled: cfg!(feature = "desktop-control"),
                    env_enabled: true,
                    startup_cost: CapabilityId::DesktopControl.startup_cost(),
                    note: "Native desktop click/type automation",
                },
                CapabilityStatus {
                    id: CapabilityId::ExecutionTools,
                    compiled: cfg!(feature = "execution-tools"),
                    env_enabled: true,
                    startup_cost: CapabilityId::ExecutionTools.startup_cost(),
                    note: "Document, vector, finance, and external execution tools",
                },
                CapabilityStatus {
                    id: CapabilityId::McpBridge,
                    compiled: cfg!(feature = "mcp-bridge"),
                    env_enabled: true,
                    startup_cost: CapabilityId::McpBridge.startup_cost(),
                    note: "External MCP adapter over Hera runtime",
                },
            ],
        }
    }

    pub fn runtime_enabled(&self, capability: CapabilityId) -> bool {
        self.statuses
            .iter()
            .find(|status| status.id == capability)
            .map(|status| status.compiled && status.env_enabled)
            .unwrap_or(false)
    }

    pub fn statuses(&self) -> &[CapabilityStatus] {
        &self.statuses
    }

    pub fn log_summary(&self) {
        for status in &self.statuses {
            info!(
                capability = status.id.as_str(),
                compiled = status.compiled,
                env_enabled = status.env_enabled,
                startup_cost = status.startup_cost,
                note = status.note,
                "Hera capability detected"
            );
        }
    }
}

fn env_flag(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}
