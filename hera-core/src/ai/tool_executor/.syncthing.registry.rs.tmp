//! Tool registry bootstrap and canonical app definitions

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use super::ToolRuntimeMetadata;
use super::security::parse_tool_risk_level;

static TOOL_RUNTIME_REGISTRY: OnceLock<HashMap<String, ToolRuntimeMetadata>> = OnceLock::new();
static REGISTERED_TOOL_NAMES: OnceLock<HashSet<String>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct CanonicalAppEntry {
    pub slug: String,
    pub path: String,
    pub manifest: String,
}

pub fn load_canonical_app_registry() -> Vec<CanonicalAppEntry> {
    let path = "/home/paulo/Programs/apps/OS/etc/apps.toml";
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };

    let mut apps = Vec::new();
    let mut current_slug: Option<String> = None;
    let mut current_path: Option<String> = None;
    let mut current_manifest: Option<String> = None;

    let flush_current = |apps: &mut Vec<CanonicalAppEntry>,
                         slug: &mut Option<String>,
                         path: &mut Option<String>,
                         manifest: &mut Option<String>| {
        if let (Some(slug), Some(path), Some(manifest)) =
            (slug.take(), path.take(), manifest.take())
        {
            apps.push(CanonicalAppEntry {
                slug,
                path,
                manifest,
            });
        }
    };

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[apps]]" {
            flush_current(
                &mut apps,
                &mut current_slug,
                &mut current_path,
                &mut current_manifest,
            );
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').to_string();
        match key {
            "slug" => current_slug = Some(value),
            "path" => current_path = Some(value),
            "manifest" => current_manifest = Some(value),
            _ => {}
        }
    }
    flush_current(
        &mut apps,
        &mut current_slug,
        &mut current_path,
        &mut current_manifest,
    );
    apps
}

pub(super) fn alias_terms_for_app(entry: &CanonicalAppEntry) -> Vec<String> {
    let mut aliases = BTreeSet::new();
    let slug = entry.slug.to_lowercase();
    aliases.insert(slug.clone());
    aliases.insert(slug.replace('-', " "));

    if let Some(last) = entry.path.split('/').next_back() {
        let lowered = last.to_lowercase();
        aliases.insert(lowered.clone());
        aliases.insert(lowered.replace('-', " "));
    }

    if let Some(last) = entry.manifest.split('/').next_back() {
        let lowered = last.trim_end_matches(".toml").to_lowercase();
        aliases.insert(lowered);
    }

    match entry.slug.as_str() {
        "latinos" => {
            aliases.insert("latinos-rust".to_string());
        }
        "vetra" => {
            aliases.insert("vetra-rust".to_string());
        }
        "movilo" => {
            aliases.insert("movilo-v3".to_string());
            aliases.insert("movilo-prod".to_string());
        }
        "os-v3" => {
            aliases.insert("os".to_string());
            aliases.insert("portal".to_string());
            aliases.insert("os-portal".to_string());
        }
        "desktop" => {
            aliases.insert("desktop-rust".to_string());
        }
        "paulo-vila-rust" => {
            aliases.insert("paulo vila".to_string());
            aliases.insert("paulovila".to_string());
            aliases.insert("paulo-vila".to_string());
        }
        "capacita" => {
            aliases.insert("capacita-rust".to_string());
        }
        "construvendo" => {
            aliases.insert("construvendo-rust".to_string());
            aliases.insert("olave bay".to_string());
            aliases.insert("olave bay tower".to_string());
        }
        _ => {}
    }

    aliases.into_iter().collect()
}

pub fn canonicalize_app_slug(input: &str) -> Option<String> {
    let needle = input.trim().to_lowercase();
    if needle.is_empty() {
        return None;
    }

    for entry in load_canonical_app_registry() {
        let aliases = alias_terms_for_app(&entry);
        if aliases.iter().any(|alias| alias == &needle) {
            return Some(entry.slug);
        }
    }
    None
}

pub fn canonical_app_search_terms(input: &str) -> Vec<String> {
    let canonical = canonicalize_app_slug(input).unwrap_or_else(|| input.trim().to_lowercase());
    let mut terms = BTreeSet::new();

    if let Some(entry) = load_canonical_app_registry()
        .into_iter()
        .find(|entry| entry.slug == canonical)
    {
        for alias in alias_terms_for_app(&entry) {
            if !alias.is_empty() {
                terms.insert(alias);
            }
        }
        if let Some(last) = entry.path.split('/').next_back() {
            terms.insert(last.to_lowercase().replace('_', "-"));
        }
    } else if !canonical.is_empty() {
        terms.insert(canonical);
    }

    terms.into_iter().collect()
}

pub fn text_contains_app_alias(text: &str, aliases: &[String]) -> bool {
    let lower = text.to_lowercase();
    aliases.iter().any(|alias| lower.contains(alias))
}

pub fn pm2_process_name_for_slug(slug: &str) -> &str {
    match slug {
        "acciona" => "acciona-rust",
        "cartera" => "cartera-rust",
        "vetra" => "vetra-rust",
        "movilo" => "movilo",
        "latinos" => "latinos-rust",
        "os-v3" => "os-v3",
        "desktop" => "desktop-rust",
        "paulo-vila-rust" => "paulo-vila",
        "capacita" => "capacita-rust",
        "construvendo" => "construvendo-rust",
        "hera" => "hera-core",
        "whisper" => "hera-core",
        "audio-stt" => "hera-core",
        "audio-engine" => "hera-core",
        "argus" => "argus",
        "memento" => "memento-node",
        _ => slug,
    }
}

fn collect_tool_runtime_metadata_in_dir(
    dir: &Path,
    registry: &mut HashMap<String, ToolRuntimeMetadata>,
) {
    if !dir.exists() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_tool_runtime_metadata_in_dir(&entry_path, registry);
            continue;
        }

        if entry_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        let content = match std::fs::read_to_string(&entry_path) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let schema = match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(schema) => schema,
            Err(_) => continue,
        };
        let Some(tool_name) = schema
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(|value| value.as_str())
        else {
            continue;
        };
        let metadata = ToolRuntimeMetadata {
            execution_kind: schema
                .get("metadata")
                .and_then(|value| value.get("execution_kind"))
                .and_then(|value| value.as_str())
                .map(ToString::to_string),
            risk_level: schema
                .get("metadata")
                .and_then(|value| value.get("risk_level"))
                .and_then(|value| value.as_str())
                .and_then(parse_tool_risk_level),
            timeout_ms: schema
                .get("metadata")
                .and_then(|value| value.get("timeout_ms"))
                .and_then(|value| value.as_u64()),
            allowed_callers: schema
                .get("metadata")
                .and_then(|value| value.get("allowed_callers"))
                .and_then(|value| value.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
        };
        if metadata.execution_kind.is_some() {
            registry.insert(tool_name.to_string(), metadata);
        }
    }
}

fn collect_tool_names_in_dir(dir: &Path, names: &mut HashSet<String>) {
    if !dir.exists() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_tool_names_in_dir(&entry_path, names);
            continue;
        }
        if entry_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let content = match std::fs::read_to_string(&entry_path) {
            Ok(content) => content,
            Err(_) => continue,
        };
        let schema = match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(schema) => schema,
            Err(_) => continue,
        };
        if let Some(tool_name) = schema
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(|value| value.as_str())
        {
            names.insert(tool_name.to_string());
        }
    }
}

pub(crate) fn tool_runtime_registry() -> &'static HashMap<String, ToolRuntimeMetadata> {
    TOOL_RUNTIME_REGISTRY.get_or_init(|| {
        let mut registry = HashMap::new();
        collect_tool_runtime_metadata_in_dir(
            Path::new("/home/paulo/Programs/apps/OS/Tools"),
            &mut registry,
        );
        registry
    })
}

pub(crate) fn find_tool_runtime_metadata(tool_name: &str) -> Option<&'static ToolRuntimeMetadata> {
    tool_runtime_registry().get(tool_name)
}

pub(crate) fn registered_tool_names() -> &'static HashSet<String> {
    REGISTERED_TOOL_NAMES.get_or_init(|| {
        let mut names = HashSet::new();
        collect_tool_names_in_dir(Path::new("/home/paulo/Programs/apps/OS/Tools"), &mut names);
        names
    })
}

pub(crate) fn is_registered_tool(tool_name: &str) -> bool {
    registered_tool_names().contains(tool_name)
}
