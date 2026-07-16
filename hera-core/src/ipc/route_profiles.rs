use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct RouteProfile {
    pub id: &'static str,
    pub app: &'static str,
    pub persona_path: &'static str,
    pub default_context_budget_mode: &'static str,
    pub prefer_stream: bool,
    pub target_p95_ms: u64,
    pub target_first_token_ms: Option<u64>,
}

const DEFAULT_PERSONA: &str = "/home/paulo/Programs/apps/OS/Agents/ava.md";

const ROUTE_PROFILES: &[RouteProfile] = &[
    RouteProfile {
        id: "cartera_widget",
        app: "cartera",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/hada_financiera.md",
        default_context_budget_mode: "lightweight",
        prefer_stream: true,
        target_p95_ms: 700,
        target_first_token_ms: Some(40),
    },
    RouteProfile {
        id: "cartera_admin_chat",
        app: "cartera",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/hada_financiera.md",
        default_context_budget_mode: "standard",
        prefer_stream: false,
        target_p95_ms: 2000,
        target_first_token_ms: Some(80),
    },
    RouteProfile {
        id: "movilo_widget",
        app: "movilo",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/memo.md",
        default_context_budget_mode: "lightweight",
        prefer_stream: true,
        target_p95_ms: 1400,
        target_first_token_ms: Some(40),
    },
    RouteProfile {
        id: "consulting_widget",
        app: "consulting",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/max.md",
        default_context_budget_mode: "lightweight",
        prefer_stream: true,
        target_p95_ms: 800,
        target_first_token_ms: Some(40),
    },
    RouteProfile {
        id: "latinos_widget",
        app: "latinos",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/latinos.md",
        default_context_budget_mode: "standard",
        prefer_stream: true,
        target_p95_ms: 900,
        target_first_token_ms: Some(40),
    },
    RouteProfile {
        id: "vetra_widget",
        app: "vetra",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/vetra.md",
        default_context_budget_mode: "standard",
        prefer_stream: true,
        target_p95_ms: 1600,
        target_first_token_ms: Some(60),
    },
    RouteProfile {
        id: "os_v3_chat",
        app: "os-v3",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/ava.md",
        default_context_budget_mode: "standard",
        prefer_stream: true,
        target_p95_ms: 1600,
        target_first_token_ms: Some(60),
    },
    RouteProfile {
        id: "hera_internal",
        app: "hera",
        persona_path: DEFAULT_PERSONA,
        default_context_budget_mode: "standard",
        prefer_stream: false,
        target_p95_ms: 1200,
        target_first_token_ms: None,
    },
    RouteProfile {
        id: "ops",
        app: "ops",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/ava_ops.md",
        // heavy = full tools + schema + memory; allow_tools=true so the agentic
        // loop engages (diagnose → read logs → propose/verify). Operator OPS
        // copilot surface (CLI `claude --ops`).
        default_context_budget_mode: "heavy",
        prefer_stream: false,
        target_p95_ms: 120_000,
        target_first_token_ms: None,
    },
    RouteProfile {
        id: "coding",
        app: "coding",
        persona_path: "/home/paulo/Programs/apps/OS/Agents/ava_coder.md",
        // heavy = full tools + schema + memory; allow_tools=true so the agentic
        // loop engages. This is the dedicated coding surface (CLI `claude`,
        // ava_coder bot) that gets low deterministic temperature.
        default_context_budget_mode: "heavy",
        prefer_stream: false,
        // Coding tasks run many tool rounds (read→edit→build→fix); generous SLO.
        target_p95_ms: 120_000,
        target_first_token_ms: None,
    },
    RouteProfile {
        id: "ava",
        app: "ava",
        persona_path: DEFAULT_PERSONA,
        // heavy = full tools + schema + memory; allow_tools=true so the agentic
        // loop engages. Ava (operator-only, Telegram+WhatsApp) was fused with
        // AvaCoder's toolset 2026-07-16 (bots.toml now grants edit_file/
        // grep_search/cargo_*) so Paulo has one bot that both chats and makes
        // real repo edits. Without this entry `profile_for_app("ava")` fell
        // through to "default" (standard budget, no agentic loop) and the
        // coding tools listed in bots.toml were silently inert.
        default_context_budget_mode: "heavy",
        prefer_stream: false,
        target_p95_ms: 120_000,
        target_first_token_ms: None,
    },
    RouteProfile {
        id: "heavy",
        app: "",
        persona_path: DEFAULT_PERSONA,
        // heavy = full tools + schema + memory; targets complex analytical tasks
        // (financial analysis, multi-step reasoning) where the primary 35B model
        // is needed. Callers can request this via route_profile: "heavy" from MCP.
        default_context_budget_mode: "heavy",
        prefer_stream: false,
        target_p95_ms: 60_000,
        target_first_token_ms: None,
    },
    RouteProfile {
        id: "claude_code",
        app: "claude_code",
        persona_path: DEFAULT_PERSONA,
        // minimal = memory ON, tool schemas + db schema OFF, allow_tools=false.
        // Claude Code calls generate_text for pure drafting / mechanical gen and
        // has its OWN tools — injecting Hera's 90-tool schema (~28K) + a DB schema
        // (~12K) was pure waste on every call (runtime telemetry flagged
        // tool_schema_inflation + db_schema_inflation for app_id=claude_code).
        // Callers that genuinely want Hera's tool loop pass route_profile
        // "coding"/"heavy" or explicit permissions/context_budget_mode, which still
        // escalate (parse_context_budget honours explicit modes; coding/heavy map to
        // heavy), so this only trims the default pure-generation call.
        default_context_budget_mode: "minimal",
        prefer_stream: false,
        target_p95_ms: 30_000,
        target_first_token_ms: None,
    },
    RouteProfile {
        id: "default",
        app: "",
        persona_path: DEFAULT_PERSONA,
        default_context_budget_mode: "standard",
        prefer_stream: false,
        target_p95_ms: 1800,
        target_first_token_ms: Some(80),
    },
];

pub fn find_route_profile(profile_id: &str) -> Option<RouteProfile> {
    ROUTE_PROFILES
        .iter()
        .find(|profile| profile.id == profile_id)
        .cloned()
}

pub fn profile_for_app(app: &str) -> RouteProfile {
    let normalized = app.trim().to_ascii_lowercase();
    ROUTE_PROFILES
        .iter()
        .find(|profile| !profile.app.is_empty() && profile.app == normalized)
        .cloned()
        .unwrap_or_else(|| {
            ROUTE_PROFILES
                .iter()
                .find(|profile| profile.id == "default")
                .expect("default route profile must exist")
                .clone()
        })
}

pub fn resolve_route_profile(explicit_profile: Option<&str>, app: &str) -> RouteProfile {
    explicit_profile
        .and_then(find_route_profile)
        .unwrap_or_else(|| profile_for_app(app))
}

pub fn all_route_profiles() -> Vec<RouteProfile> {
    ROUTE_PROFILES.to_vec()
}
