//! Handler: download_lora (CivitAI LoRA management).

use super::types::{HandlerOutcome, IpcPayload};

/// Handle the "download_lora" action — download LoRA from CivitAI or direct URL.
pub async fn handle_download_lora(request: &IpcPayload) -> HandlerOutcome {
    let url = request
        .payload
        .get("url")
        .and_then(|u| u.as_str())
        .unwrap_or("")
        .to_string();
    let custom_name = request
        .payload
        .get("name")
        .and_then(|n| n.as_str())
        .map(|s| s.to_string());
    let lora_dir = "/home/paulo/models/image-stack/loras";

    // Read CivitAI token from Imaginclaw .env
    let civitai_token =
        std::fs::read_to_string("/home/paulo/Programs/apps/OS/Imaginclaw/.env")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("CIVITAI_TOKEN"))
                    .map(|l| {
                        l.split('=')
                            .nth(1)
                            .unwrap_or("")
                            .trim_matches('"')
                            .to_string()
                    })
            })
            .unwrap_or_default();

    if url.is_empty() {
        return HandlerOutcome::Result {
            result_text: "❌ No URL provided".to_string(),
            origin: "unknown".to_string(),
            model: String::new(),
            tool_calls: None,
        };
    }
    if civitai_token.is_empty() {
        return HandlerOutcome::Result {
            result_text: "❌ CIVITAI_TOKEN not found in .env".to_string(),
            origin: "unknown".to_string(),
            model: String::new(),
            tool_calls: None,
        };
    }

    // Check if it's a CivitAI page URL
    let model_id_opt: Option<String> = if url.contains("civitai.com/models/") {
        url.split("civitai.com/models/")
            .nth(1)
            .and_then(|s| s.split('/').next())
            .map(|s| s.to_string())
    } else {
        None
    };

    let result_text = if let Some(model_id) = model_id_opt {
        download_from_civitai(&model_id, &civitai_token, custom_name, lora_dir).await
    } else {
        download_direct(&url, custom_name, lora_dir).await
    };

    HandlerOutcome::Result {
        result_text,
        origin: "unknown".to_string(),
        model: String::new(),
        tool_calls: None,
    }
}

/// Download a LoRA from CivitAI API using model ID.
async fn download_from_civitai(
    model_id: &str,
    civitai_token: &str,
    custom_name: Option<String>,
    lora_dir: &str,
) -> String {
    tracing::info!(
        "📦 Fetching CivitAI model metadata for ID: {}",
        model_id
    );
    let api_url = format!("https://civitai.com/api/v1/models/{}", model_id);
    let client = reqwest::Client::new();
    match client
        .get(&api_url)
        .header("Authorization", format!("Bearer {}", civitai_token))
        .send()
        .await
    {
        Ok(api_resp) if api_resp.status().is_success() => {
            match api_resp.json::<serde_json::Value>().await {
                Ok(model_data) => {
                    let versions = model_data
                        .get("modelVersions")
                        .and_then(|v| v.as_array());
                    let first_version = versions.and_then(|arr| arr.first());
                    let files = first_version
                        .and_then(|ver| ver.get("files"))
                        .and_then(|f| f.as_array());
                    let safetensors_file = files.and_then(|arr| {
                        arr.iter().find(|f| {
                            f.get("name")
                                .and_then(|n| n.as_str())
                                .map(|n| n.ends_with(".safetensors"))
                                .unwrap_or(false)
                        })
                    });

                    let download_url = safetensors_file
                        .and_then(|f| f.get("downloadUrl"))
                        .and_then(|u| u.as_str());
                    let file_name = safetensors_file
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("lora.safetensors");

                    let final_name = custom_name
                        .map(|n| {
                            if n.ends_with(".safetensors") {
                                n
                            } else {
                                format!("{}.safetensors", n)
                            }
                        })
                        .unwrap_or_else(|| file_name.to_string());

                    if let Some(dl_url) = download_url {
                        tracing::info!("⬇️ Downloading LoRA: {} -> {}", dl_url, final_name);
                        let dest_path = format!("{}/{}", lora_dir, final_name);
                        match client
                            .get(dl_url)
                            .header(
                                "Authorization",
                                format!("Bearer {}", civitai_token),
                            )
                            .send()
                            .await
                        {
                            Ok(dl_resp) if dl_resp.status().is_success() => {
                                match dl_resp.bytes().await {
                                    Ok(bytes) => match std::fs::write(&dest_path, &bytes) {
                                        Ok(_) => {
                                            let size_mb =
                                                bytes.len() as f64 / 1_048_576.0;
                                            let tag_name = final_name
                                                .trim_end_matches(".safetensors")
                                                .to_string();

                                            let auto_tags = extract_auto_tags(
                                                &model_data,
                                                &tag_name,
                                            );
                                            update_triggers_json(
                                                lora_dir,
                                                &tag_name,
                                                &auto_tags,
                                            );

                                            tracing::info!(
                                                "✅ LoRA saved: {} ({:.1} MB)",
                                                dest_path,
                                                size_mb
                                            );
                                            format!(
                                                "✅ LoRA downloaded: {} ({:.1} MB)\n⚡️ Auto-Triggers extracted: {}",
                                                final_name, size_mb, auto_tags.join(", ")
                                            )
                                        }
                                        Err(e) => format!("❌ Failed to save file: {}", e),
                                    },
                                    Err(e) => {
                                        format!("❌ Download stream error: {}", e)
                                    }
                                }
                            }
                            Ok(dl_resp) => {
                                format!("❌ Download failed: HTTP {}", dl_resp.status())
                            }
                            Err(e) => {
                                format!("❌ Download request failed: {}", e)
                            }
                        }
                    } else {
                        "❌ No .safetensors file found in model versions".to_string()
                    }
                }
                Err(e) => {
                    format!("❌ Failed to parse CivitAI API response: {}", e)
                }
            }
        }
        Ok(api_resp) => {
            format!("❌ CivitAI API error: HTTP {}", api_resp.status())
        }
        Err(e) => {
            format!("❌ CivitAI API request failed: {}", e)
        }
    }
}

/// Download a LoRA from a direct URL (non-CivitAI).
async fn download_direct(
    url: &str,
    custom_name: Option<String>,
    lora_dir: &str,
) -> String {
    let final_name =
        custom_name.unwrap_or_else(|| "downloaded_lora.safetensors".to_string());
    let dest_path = format!("{}/{}", lora_dir, final_name);
    let client = reqwest::Client::new();
    match client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => match resp.bytes().await {
            Ok(bytes) => match std::fs::write(&dest_path, &bytes) {
                Ok(_) => {
                    let size_mb = bytes.len() as f64 / 1_048_576.0;
                    format!("✅ LoRA downloaded: {} ({:.1} MB)", final_name, size_mb)
                }
                Err(e) => format!("❌ Failed to save: {}", e),
            },
            Err(e) => format!("❌ Download error: {}", e),
        },
        Ok(resp) => format!("❌ HTTP {}", resp.status()),
        Err(e) => format!("❌ Request failed: {}", e),
    }
}

/// Extract trained words and model title from CivitAI model data.
fn extract_auto_tags(
    model_data: &serde_json::Value,
    tag_name: &str,
) -> Vec<String> {
    let mut auto_tags = Vec::new();
    if let Some(trained) = model_data
        .get("modelVersions")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.get("trainedWords"))
        .and_then(|w| w.as_array())
    {
        for t in trained {
            if let Some(ts) = t.as_str() {
                auto_tags.push(ts.to_string());
            }
        }
    }
    if let Some(title) = model_data.get("name").and_then(|n| n.as_str()) {
        auto_tags.push(title.to_string());
    }
    if auto_tags.is_empty() {
        auto_tags.push(tag_name.to_string());
    }
    auto_tags
}

/// Append or update the LoRA auto-trigger entry in triggers.json.
fn update_triggers_json(lora_dir: &str, tag_name: &str, auto_tags: &[String]) {
    let triggers_path = format!("{}/triggers.json", lora_dir);
    if let Ok(content) = std::fs::read_to_string(&triggers_path)
        && let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&content)
        && let Some(obj) = json.as_object_mut()
    {
        obj.insert(tag_name.to_string(), serde_json::json!(auto_tags));
        if let Ok(new_content) = serde_json::to_string_pretty(&json) {
            let _ = std::fs::write(&triggers_path, new_content);
        }
    }
}
