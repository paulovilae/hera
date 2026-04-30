use anyhow::{Result, anyhow};
use crate::mcp::client::McpHttpClient;
use serde_json::json;
use base64::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Hera is the primary Multimodal Rust Agent orchestrator inside the Execution Layer.
/// She natively binds the text-only neural inference loops to the rich Canvas capabilities
/// (Image, Audio, Video generation) via direct API bounds or MCP.
pub struct Hera {
    pub mcp_client: McpHttpClient,
    pub http_client: reqwest::Client,
    pub draw_url: String,
}

fn generated_outputs_dir() -> Result<PathBuf> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| anyhow!("failed to resolve ImagineOS repo root"))?;
    Ok(repo_root.join("Hera/playground/outputs"))
}

/// Represents a checkpoint model available for image generation.
#[derive(serde::Serialize, Clone)]
pub struct CheckpointEntry {
    pub filename: String,
    pub name: String,
    pub family: String,
}

/// Represents a LoRA adapter available for image generation.
#[derive(serde::Serialize, Clone)]
pub struct LoraEntry {
    pub filename: String,
    pub name: String,
    pub trigger_words: Vec<String>,
    pub base_model: String,
}

impl Hera {
    /// Mounts Hera to the Sovereign OS routing bounds.
    pub fn new(smartos_router_url: &str) -> Self {
        Self {
            mcp_client: McpHttpClient::new(smartos_router_url),
            http_client: reqwest::Client::new(),
            draw_url: "http://127.0.0.1:8999".to_string(),
        }
    }

    fn detect_family(filename: &str) -> String {
        let lower = filename.to_lowercase();
        if lower.contains("flux") { "flux".into() }
        else if lower.contains("pony") { "pony".into() }
        else if lower.contains("ltx") || lower.contains("wan") { "video".into() }
        else if lower.contains("sdxl") || lower.contains("xl") || lower.contains("illustrious") { "sdxl".into() }
        else if lower.contains("sana") { "sana".into() }
        else if lower.contains("1.5") || lower.contains("sd15") { "sd15".into() }
        else { "sdxl".into() }
    }

    fn clean_model_name(filename: &str) -> String {
        filename
            .replace(".safetensors", "")
            .replace(".ckpt", "")
            .replace(".gguf", "")
            .replace('_', " ")
            .replace(" - ", " — ")
    }

    pub fn list_checkpoints() -> Vec<CheckpointEntry> {
        let dir = "/data/models/swarmui/Stable-Diffusion";
        let mut entries = Vec::new();

        if let Ok(read_dir) = fs::read_dir(dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                let filename = entry.file_name().to_string_lossy().to_string();
                if path.is_dir() { continue; }
                if !filename.ends_with(".safetensors")
                    && !filename.ends_with(".ckpt")
                    && !filename.ends_with(".gguf")
                {
                    continue;
                }

                entries.push(CheckpointEntry {
                    name: Self::clean_model_name(&filename),
                    family: Self::detect_family(&filename),
                    filename,
                });
            }
        }

        let flux_dir = format!("{}/Flux", dir);
        if let Ok(read_dir) = fs::read_dir(&flux_dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                let filename = entry.file_name().to_string_lossy().to_string();
                if path.is_dir() || (!filename.ends_with(".safetensors") && !filename.ends_with(".gguf")) {
                    continue;
                }
                entries.push(CheckpointEntry {
                    name: Self::clean_model_name(&filename),
                    family: "flux".into(),
                    filename: format!("Flux/{}", filename),
                });
            }
        }

        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    pub fn list_loras() -> Vec<LoraEntry> {
        let dir = "/data/models/swarmui/Lora";
        let mut entries = Vec::new();
        Self::scan_lora_dir(dir, "", &mut entries);
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    fn scan_lora_dir(dir: &str, prefix: &str, entries: &mut Vec<LoraEntry>) {
        let read_dir = match fs::read_dir(dir) {
            Ok(d) => d,
            Err(_) => return,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            let filename = entry.file_name().to_string_lossy().to_string();

            if path.is_dir() {
                let new_prefix = if prefix.is_empty() {
                    filename.clone()
                } else {
                    format!("{}/{}", prefix, filename)
                };
                Self::scan_lora_dir(&path.to_string_lossy(), &new_prefix, entries);
                continue;
            }

            if !filename.ends_with(".safetensors") && !filename.ends_with(".pt") {
                continue;
            }

            let base_path = path.to_string_lossy().replace(".safetensors", "").replace(".pt", "");
            let json_path = format!("{}.json", base_path);

            let (trigger_words, base_model) = if Path::new(&json_path).exists() {
                Self::parse_lora_sidecar(&json_path)
            } else {
                (vec![], "Unknown".into())
            };

            let full_filename = if prefix.is_empty() {
                filename.clone()
            } else {
                format!("{}/{}", prefix, filename)
            };

            entries.push(LoraEntry {
                name: Self::clean_model_name(&filename),
                trigger_words,
                base_model,
                filename: full_filename,
            });
        }
    }

    fn parse_lora_sidecar(json_path: &str) -> (Vec<String>, String) {
        let content = match fs::read_to_string(json_path) {
            Ok(c) => c,
            Err(_) => return (vec![], "Unknown".into()),
        };

        let parsed: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return (vec![], "Unknown".into()),
        };

        let base_model = parsed.get("baseModel")
            .or_else(|| parsed.get("base_model"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let trigger_words = if let Some(words) = parsed.get("trainedWords").and_then(|v| v.as_array()) {
            words.iter().filter_map(|w| w.as_str().map(|s| s.to_string())).collect()
        } else if let Some(activation) = parsed.get("activation_text").and_then(|v| v.as_str()) {
            activation.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
        } else {
            vec![]
        };

        (trigger_words, base_model)
    }

    pub async fn describe_sketch(&self, image_data_uri: &str) -> Result<String> {
        let vision_url = "http://127.0.0.1:3305/v1/chat/completions";
        let payload = json!({
            "model": "hera",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert image prompt engineer. Given a sketch or drawing, describe WHAT is depicted in vivid detail suitable for an AI image generator. Focus on: subject, pose, style, colors, mood, background. Output ONLY the prompt, no preamble, no quotes. Keep it under 80 words. Make it painterly and beautiful."
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Describe this sketch for an AI image generator:" },
                        { "type": "image_url", "image_url": { "url": image_data_uri } }
                    ]
                }
            ],
            "stream": false,
            "temperature": 0.7,
            "max_tokens": 150
        });

        let res = self.http_client.post(vision_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !res.status().is_success() {
            let err = res.text().await.unwrap_or_default();
            return Err(anyhow!("Vision LLM failed: {}", err));
        }

        let result: serde_json::Value = res.json().await?;
        let description = result.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("A beautiful, highly detailed, professional illustration. High quality masterpiece.")
            .trim()
            .to_string();

        Ok(description)
    }

    pub async fn generate_image(
        &self,
        prompt: &str,
        _engine: Option<&str>,
        width: Option<u32>,
        height: Option<u32>,
        steps: Option<u32>,
        model: Option<&str>,
        loras: Option<&Vec<serde_json::Value>>,
        init_image: Option<&str>,
        denoising_strength: Option<f32>,
        cfg_scale: Option<f32>,
        _nsfw: Option<bool>,
    ) -> Result<serde_json::Value> {
        let effective_prompt = if prompt.trim().is_empty() {
            if let Some(img) = init_image {
                match self.describe_sketch(img).await {
                    Ok(desc) => desc,
                    Err(_) => "A beautiful, highly detailed, professional illustration based on the sketch. High quality masterpiece.".to_string(),
                }
            } else {
                prompt.to_string()
            }
        } else {
            prompt.to_string()
        };
        let w = width.unwrap_or(512);
        let h = height.unwrap_or(512);
        let use_img2img = init_image.is_some();
        let endpoint = if use_img2img {
            format!("{}/sdapi/v1/img2img", self.draw_url)
        } else {
            format!("{}/sdapi/v1/txt2img", self.draw_url)
        };

        let mut payload = json!({
            "prompt": effective_prompt,
            "width": w,
            "height": h,
            "seed": rand::random::<u32>()
        });

        if let Some(s) = steps {
            payload.as_object_mut().unwrap().insert("steps".to_string(), json!(s));
        }
        if let Some(m) = model {
            payload.as_object_mut().unwrap().insert("override_settings".to_string(), json!({
                "sd_model_checkpoint": m
            }));
        }
        if let Some(lora_list) = loras {
            if !lora_list.is_empty() {
                payload.as_object_mut().unwrap().insert("loras".to_string(), json!(lora_list));
            }
        }
        if let Some(cfg) = cfg_scale {
            payload.as_object_mut().unwrap().insert("cfg_scale".to_string(), json!(cfg));
        }
        if let Some(img) = init_image {
            payload.as_object_mut().unwrap().insert("init_images".to_string(), json!([img]));
            payload.as_object_mut().unwrap().insert(
                "denoising_strength".to_string(),
                json!(denoising_strength.unwrap_or(0.65)),
            );
        }

        let res = self.http_client.post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !res.status().is_success() {
            let error_text = res.text().await?;
            return Err(anyhow!("SwarmUI Generation Failed: {}", error_text));
        }

        let result_json: serde_json::Value = res.json().await?;
        let b64_result = result_json.get("images")
            .and_then(|images| images.as_array())
            .and_then(|arr| arr.first())
            .and_then(|img| img.as_str())
            .map(|s| s.to_string());

        if let Some(b64) = b64_result {
            let clean_b64 = if b64.starts_with("data:image") {
                b64.split(',').nth(1).unwrap_or(&b64).to_string()
            } else {
                b64
            };

            let image_data = BASE64_STANDARD.decode(&clean_b64)?;
            let output_dir = generated_outputs_dir()?;
            fs::create_dir_all(&output_dir)?;

            let filename = format!("hera_drawn_{}.png", Uuid::new_v4());
            let filepath = output_dir.join(&filename);
            fs::write(&filepath, image_data)?;

            return Ok(json!({
                "status": "success",
                "image_url": format!("/outputs/{}", filename),
                "url": format!("data:image/png;base64,{}", clean_b64),
                "auto_prompt": effective_prompt
            }));
        }

        Err(anyhow!("Invalid response format from SwarmUI"))
    }

    pub async fn synthesize_speech(&self, text: &str, voice: Option<&str>) -> Result<serde_json::Value> {
        let _ = self.mcp_client.initialize().await;
        let mut args = json!({ "text": text });
        if let Some(v) = voice {
            args.as_object_mut().unwrap().insert("voice".to_string(), json!(v));
        }
        let res = self.mcp_client.call_tool("smartos_speak", args).await?;
        Ok(serde_json::to_value(res)?)
    }

    pub async fn synthesize_video(&self, prompt: &str) -> Result<serde_json::Value> {
        let _ = self.mcp_client.initialize().await;
        let args = json!({
            "prompt": prompt,
            "width": 768,
            "height": 512,
            "num_frames": 97
        });
        let res = self.mcp_client.call_tool("smartos_ltx2", args).await?;
        Ok(serde_json::to_value(res)?)
    }

    pub async fn native_web_scrape(&self, url: &str) -> Result<String> {
        let res = self.http_client.get(url).send().await?;
        if !res.status().is_success() {
            return Err(anyhow!("Failed to fetch URL: {}", res.status()));
        }
        let html_content = res.text().await?;
        let document = scraper::Html::parse_document(&html_content);

        let mut text_blocks = Vec::new();
        for node in document.tree.nodes() {
            if let scraper::node::Node::Element(el) = node.value() {
                let tag = el.name();
                if tag != "script" && tag != "style" && tag != "noscript" && tag != "svg" && tag != "path" {
                }
            } else if let scraper::node::Node::Text(text) = node.value() {
                let trimmed = text.text.trim();
                let parent = node.parent().and_then(|p| p.value().as_element());
                if let Some(pel) = parent {
                    let ptag = pel.name();
                    if ptag != "script" && ptag != "style" && ptag != "noscript" && ptag != "svg" && ptag != "path" {
                        if !trimmed.is_empty() {
                            text_blocks.push(trimmed.to_string());
                        }
                    }
                }
            }
        }
        text_blocks.dedup();
        let full_text = text_blocks.join(" ");
        let max_len = 16000;
        let truncated = if full_text.len() > max_len {
            format!("{}... (truncated)", &full_text[..max_len])
        } else {
            full_text
        };
        Ok(truncated)
    }

    pub async fn native_web_search(&self, query: &str) -> Result<String> {
        let url = "https://lite.duckduckgo.com/lite/";
        let params = [("q", query)];
        
        let res = self.http_client.post(url)
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            .form(&params)
            .send().await?;
            
        if !res.status().is_success() {
            return Err(anyhow!("Search failed: {}", res.status()));
        }
        let html_content = res.text().await?;
        let document = scraper::Html::parse_document(&html_content);
        
        let link_selector = scraper::Selector::parse(".result-link").unwrap();
        let snippet_selector = scraper::Selector::parse(".result-snippet").unwrap();

        let links: Vec<_> = document.select(&link_selector).collect();
        let snippets: Vec<_> = document.select(&snippet_selector).collect();

        let mut results = Vec::new();
        let count = links.len().min(snippets.len());
        for i in 0..count.min(5) {
            let title = links[i].text().collect::<Vec<_>>().join(" ").trim().to_string();
            let url = links[i].value().attr("href").unwrap_or_default().to_string();
            let snippet = snippets[i].text().collect::<Vec<_>>().join(" ").trim().to_string();
            
            if !title.is_empty() && !snippet.is_empty() {
                results.push(format!("Title: {}\nURL: {}\nSnippet: {}\n", title, url, snippet));
            }
        }

        if results.is_empty() {
            Ok("No results found.".to_string())
        } else {
            Ok(results.join("\n---\n"))
        }
    }

    pub async fn chat(&self, mut payload: serde_json::Value) -> Result<reqwest::Response> {
        let hera_persona = json!({
            "role": "system",
            "content": "You are Hera, the multimodal agent of the Hera. You can natively generate images, synthesize audio, and analyze pictures. You are extremely helpful and concise."
        });

        if let Some(messages) = payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
            messages.insert(0, hera_persona);
        }

        let chat_url = "http://127.0.0.1:3005/v1/chat/completions";
        let res = self.http_client.post(chat_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        Ok(res)
    }
}
