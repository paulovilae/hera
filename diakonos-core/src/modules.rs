use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleCategory {
    Core,
    Workflow,
    Web,
    Media,
    Docs,
    Market,
    Vector,
}

impl ModuleCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Workflow => "workflow",
            Self::Web => "web",
            Self::Media => "media",
            Self::Docs => "docs",
            Self::Market => "market",
            Self::Vector => "vector",
        }
    }

    pub fn all() -> &'static [ModuleCategory] {
        &[
            ModuleCategory::Core,
            ModuleCategory::Workflow,
            ModuleCategory::Web,
            ModuleCategory::Media,
            ModuleCategory::Docs,
            ModuleCategory::Market,
            ModuleCategory::Vector,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleConfigFile {
    #[serde(default)]
    pub enabled: Vec<ModuleCategory>,
}

#[derive(Debug, Clone)]
pub struct ModuleRegistry {
    config_path: PathBuf,
    enabled: Arc<RwLock<BTreeSet<ModuleCategory>>>,
    modified_at: Arc<RwLock<Option<SystemTime>>>,
}

impl ModuleRegistry {
    pub async fn load(config_path: impl Into<PathBuf>) -> Self {
        let config_path = config_path.into();
        let enabled = load_enabled_modules(&config_path);
        let modified_at = read_modified_time(&config_path);
        Self {
            config_path,
            enabled: Arc::new(RwLock::new(enabled)),
            modified_at: Arc::new(RwLock::new(modified_at)),
        }
    }

    pub async fn reload(&self) -> Vec<String> {
        let enabled = load_enabled_modules(&self.config_path);
        let modified_at = read_modified_time(&self.config_path);
        let mut guard = self.enabled.write().await;
        *guard = enabled;
        let mut modified_guard = self.modified_at.write().await;
        *modified_guard = modified_at;
        guard
            .iter()
            .map(|module| module.as_str().to_string())
            .collect()
    }

    pub async fn reload_if_changed(&self) -> Option<Vec<String>> {
        let current_modified = read_modified_time(&self.config_path);
        let mut modified_guard = self.modified_at.write().await;
        if *modified_guard == current_modified {
            return None;
        }
        *modified_guard = current_modified;
        drop(modified_guard);
        Some(self.reload().await)
    }

    pub async fn is_enabled(&self, category: ModuleCategory) -> bool {
        self.enabled.read().await.contains(&category)
    }

    pub async fn enabled_module_names(&self) -> Vec<String> {
        self.enabled
            .read()
            .await
            .iter()
            .map(|module| module.as_str().to_string())
            .collect()
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
}

pub fn action_module(action: &str) -> Option<ModuleCategory> {
    match action {
        "health" | "capabilities" | "list_modules" | "reload_modules" | "read_file" => Some(ModuleCategory::Core),
        "list_tools" => Some(ModuleCategory::Docs),
        "execute_dag" | "parse_dify" | "execute_workflow_proxy" => Some(ModuleCategory::Workflow),
        "web_scrape" | "web_search" => Some(ModuleCategory::Web),
        "draw_image" | "speak_text" | "generate_video" => Some(ModuleCategory::Media),
        _ => None,
    }
}

pub async fn module_snapshot(registry: &ModuleRegistry) -> BTreeMap<String, bool> {
    let enabled = registry.enabled.read().await;
    ModuleCategory::all()
        .iter()
        .map(|category| (category.as_str().to_string(), enabled.contains(category)))
        .collect()
}

fn load_enabled_modules(config_path: &Path) -> BTreeSet<ModuleCategory> {
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(parsed) = serde_json::from_str::<ModuleConfigFile>(&content) {
            return parsed.enabled.into_iter().collect();
        }
    }

    ModuleCategory::all().iter().copied().collect()
}

fn read_modified_time(config_path: &Path) -> Option<SystemTime> {
    std::fs::metadata(config_path)
        .ok()
        .and_then(|meta| meta.modified().ok())
}
