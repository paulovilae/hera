//! Media tool executors: draw, animate_avatar, speak, video, review_image, generate_music.
use crate::ai::tool_executor::{ToolCall, ToolResult};
use crate::ipc::media_safety::{self, MediaRequestContext};
use super::{hera_execution_agent, run_agent_via_hera_ipc};
use tracing::info;

/// Build a media-safety context for a tool-executor call. Identity/permissions
/// come from the optional `_hera` metadata the dispatcher attaches; when absent
/// they default to unknown/empty, which makes Tier B (NSFW) fail **closed**.
fn tool_media_ctx(call: &ToolCall, prompt: &str) -> MediaRequestContext {
    let hera = call.arguments.get("_hera");
    let caller = hera
        .and_then(|h| h.get("caller").or_else(|| h.get("app_name")).or_else(|| h.get("app")))
        .and_then(|v| v.as_str())
        .or_else(|| call.arguments.get("app").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string();
    let permissions = hera
        .and_then(|h| h.get("permissions"))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|p| p.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let chat_id = hera
        .and_then(|h| h.get("chat_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    MediaRequestContext {
        media_kind: "image".to_string(),
        requester_id: caller.clone(),
        chat_id,
        sender_name: String::new(),
        channel: "tool".to_string(),
        bot_name: caller,
        permissions,
        prompt_raw: prompt.to_string(),
        prompt_final: prompt.to_string(),
        seed: None,
        engine: "sd.cpp".to_string(),
        steps: None,
        cfg_scale: None,
    }
}

/// Music prompt enhancer — mirrors `handle_generate_video`'s "Phase 1: Brain" pattern
/// (`ipc/handler_media.rs`), the only existing automatic LLM prompt-expansion in the
/// codebase (images only have a keyword-based LoRA auto-router, not LLM enhancement).
/// Tool executors don't hold an `IpcState`/engine handle, so this goes through the
/// same self-loopback Hera IPC call already used by `run_agent_via_hera_ipc` for
/// agent personas (`execute_spawn_parallel_agents`). Best-effort: falls back to the
/// raw prompt on any failure so a slow/unavailable brain never blocks generation.
async fn enhance_music_prompt(raw_prompt: &str) -> String {
    let persona = "You are a music producer AI. Given a short brief idea, expand it into \
        a single detailed prompt for a text-to-music model (ACE-Step). Describe genre, \
        instruments, mood, tempo (BPM if it helps), and structure (intro/build/loop). \
        Do NOT use visual or image language (no cameras, lighting, colors). \
        Do not think, reason, verify, or explain your answer — do not use <think> tags. \
        Output ONLY the expanded music prompt itself, nothing else, max 2 sentences."
        .to_string();

    match run_agent_via_hera_ipc(persona, raw_prompt.to_string()).await {
        Ok(enhanced) if !enhanced.trim().is_empty() => {
            let enhanced = sanitize_enhanced_music_prompt(enhanced.trim(), raw_prompt);
            info!(
                "🎵🧠 Enhanced music prompt: {}",
                &enhanced[..enhanced.len().min(160)]
            );
            enhanced
        }
        Ok(_) => raw_prompt.to_string(),
        Err(e) => {
            tracing::warn!(
                "🎵🧠 Music prompt enhancement failed, using raw prompt: {}",
                e
            );
            raw_prompt.to_string()
        }
    }
}

/// Hard backstop against reasoning leaks the persona instruction alone doesn't
/// always stop (observed live: a ~900-char self-verification monologue —
/// "Para verificar esta descripción, necesito confirmar que el prompt cumple
/// con los requisitos..." — sent straight to ACE-Step as the style prompt).
/// `parse_ipc_result` already strips a well-formed `<think>...</think>` block;
/// this catches what's left: cap length and drop the reasoning-monologue tail
/// if one slipped through unwrapped.
fn sanitize_enhanced_music_prompt(enhanced: &str, raw_prompt: &str) -> String {
    const MAX_CHARS: usize = 400;
    // A local model narrating its own reasoning tends to open a new sentence
    // with one of these — cut there if found, keeping only the real answer.
    const LEAK_MARKERS: &[&str] = &[
        "Para verificar",
        "para verificar",
        "To verify",
        "to verify",
        "Let me verify",
        "Let me think",
        "I need to confirm",
        "necesito confirmar",
    ];
    let mut text = enhanced;
    for marker in LEAK_MARKERS {
        if let Some(pos) = text.find(marker) {
            text = text[..pos].trim_end();
        }
    }
    let text = text.trim();
    if text.is_empty() {
        return raw_prompt.to_string();
    }
    if text.len() <= MAX_CHARS {
        return text.to_string();
    }
    // Truncate at the last sentence boundary before the cap so we don't cut
    // ACE-Step's style prompt mid-word.
    let truncated = &text[..MAX_CHARS];
    match truncated.rfind(['.', '\n']) {
        Some(cut) if cut > MAX_CHARS / 3 => truncated[..=cut].trim().to_string(),
        _ => truncated.trim().to_string(),
    }
}

pub(crate) async fn execute_draw(call: &ToolCall) -> ToolResult {
    let prompt = call
        .arguments
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("A beautiful digital artwork");
    let width = call
        .arguments
        .get("width")
        .and_then(|w| w.as_u64())
        .map(|w| w as u32);
    let height = call
        .arguments
        .get("height")
        .and_then(|h| h.as_u64())
        .map(|h| h as u32);

    // Content safety gate (same gate as the /draw IPC path — Tier A illegal is
    // blocked unconditionally; Tier B NSFW needs `nsfw_allowed`, which is not
    // usually forwarded to tool calls, so autonomous NSFW draws fail closed).
    let media_ctx = tool_media_ctx(call, prompt);
    let (decision, details) = media_safety::evaluate_gate_via_ipc(&media_ctx).await;
    if decision.is_blocked() {
        media_safety::record_media_generation(&media_ctx, &decision, details.as_ref(), None, "png");
        tracing::warn!(
            "🛡️ hera_draw tool blocked ({}) caller={}",
            decision.audit_label(),
            media_ctx.bot_name
        );
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: decision.user_message(),
        };
    }

    let hera = hera_execution_agent();
    match hera
        .generate_image(
            prompt, None, width, height, None, None, None, None, None, None, None,
        )
        .await
    {
        Ok(res) => {
            let image_url = res
                .get("image_url")
                .and_then(|u| u.as_str())
                .unwrap_or("(no URL)");
            info!("🎨 [Hera] Image generated: {}", image_url);
            // Audit the allowed generation (tool path writes the blob to the
            // hera-web outputs dir; the row records prompt + identity + decision).
            media_safety::record_media_generation(&media_ctx, &decision, details.as_ref(), None, "png");

            // Build a public URL that candle-core serves at /outputs/{filename}
            // The filename is the last segment of image_url (e.g., "/outputs/hera_drawn_UUID.png")
            let filename = image_url.split('/').next_back().unwrap_or(image_url);
            let public_url = format!("https://imaginos.ai/outputs/{}", filename);
            let response = format!(
                "Image generated successfully!\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the image is delivered inline.",
                public_url
            );

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: response,
            }
        }
        Err(e) => {
            tracing::error!("🎨 [Hera] Image generation failed: {:?}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Image generation failed: {}", e),
            }
        }
    }
}

/// Extract lyrics from free-text prompt and return (style_prompt, lyrics).
/// Handles: "letra: ...", "lyrics: ...", "que diga: ...", "[Verso]...", "[Verse]..."
/// The style portion (without the lyrics clause) is returned as the prompt for the enhancer.
fn split_lyrics_from_prompt(text: &str) -> (String, Option<String>) {
    let lower = text.to_lowercase();

    // Explicit "letra:" / "lyrics:" / "que diga:" delimiter — everything after is lyrics
    for marker in &["letra:", "lyrics:", "que diga:", "que diga "] {
        if let Some(pos) = lower.find(marker) {
            let style = text[..pos].trim().to_string();
            let raw_lyrics = text[pos + marker.len()..].trim().to_string();
            if !raw_lyrics.is_empty() {
                let lyrics = raw_lyrics
                    .trim_matches(|c| c == '"' || c == '\'' || c == '\u{201c}' || c == '\u{201d}')
                    .trim()
                    .to_string();
                return (if style.is_empty() { "music".to_string() } else { style }, Some(lyrics));
            }
        }
    }
    // Structural markers anywhere in the text indicate lyric content inline
    if lower.contains("[verso]") || lower.contains("[coro]")
        || lower.contains("[verse]") || lower.contains("[chorus]")
        || lower.contains("[bridge]") || lower.contains("[puente]")
    {
        // Everything before the first structural marker = style hint; rest = lyrics
        let structural = ["[verso]", "[coro]", "[verse]", "[chorus]", "[bridge]", "[puente]"];
        if let Some(first) = structural.iter().filter_map(|m| lower.find(m)).min() {
            let style = text[..first].trim().to_string();
            let lyrics = text[first..].trim().to_string();
            return (if style.is_empty() { "music".to_string() } else { style }, Some(lyrics));
        }
    }
    (text.to_string(), None)
}

/// Parse "60 segundos", "30 seconds", "45s", "2 minutos", "2 minutes" from free text.
fn extract_duration_from_text(text: &str) -> Option<u32> {
    let lower = text.to_lowercase();
    // "2 minutos" / "2 minutes" → seconds
    let minute_pat = ["minuto", "minutos", "minute", "minutes", "min"];
    for pat in &minute_pat {
        if let Some(pos) = lower.find(pat) {
            let before = lower[..pos].trim_end();
            if let Some(n) = before.split_whitespace().next_back().and_then(|s| s.parse::<u32>().ok()) {
                return Some((n * 60).clamp(10, 120));
            }
        }
    }
    // "60 segundos" / "60 seconds" / "60s"
    let sec_pat = ["segundo", "segundos", "second", "seconds"];
    for pat in &sec_pat {
        if let Some(pos) = lower.find(pat) {
            let before = lower[..pos].trim_end();
            if let Some(n) = before.split_whitespace().next_back().and_then(|s| s.parse::<u32>().ok()) {
                return Some(n.clamp(10, 120));
            }
        }
    }
    // bare "60s" (digit immediately followed by 's')
    for word in lower.split_whitespace() {
        if word.ends_with('s') {
            if let Ok(n) = word[..word.len()-1].parse::<u32>() {
                if n >= 10 { return Some(n.clamp(10, 120)); }
            }
        }
    }
    None
}

pub(crate) async fn execute_generate_music(call: &ToolCall) -> ToolResult {
    let prompt = call
        .arguments
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("An upbeat instrumental loop");
    let duration = call
        .arguments
        .get("duration")
        .and_then(|d| d.as_u64())
        .map(|d| d as u32)
        // Fallback: extract "N segundos" / "N seconds" / "Ns" from the prompt text
        // when the model absorbed duration into text rather than as a parameter.
        .or_else(|| extract_duration_from_text(prompt));
    // Prefer explicit `lyrics` param; fall back to extracting from the raw prompt text
    // (local model often absorbs user-provided lyrics into the style description).
    let (style_prompt, extracted_lyrics) = split_lyrics_from_prompt(prompt);
    let lyrics = call
        .arguments
        .get("lyrics")
        .and_then(|l| l.as_str())
        .map(|s| s.to_string())
        .or(extracted_lyrics);

    // When there's lyrics, ACE-Step needs room for intro+verse+chorus — 10-12s
    // (the backend's own default/typical short request) isn't enough for the vocal
    // to land intelligibly. Bump the floor to 30s (still well inside the backend's
    // 10-120s range) when the requested/default duration falls short. No lyrics:
    // leave duration behavior exactly as before.
    const MIN_DURATION_WITH_LYRICS: u32 = 30;
    let mut duration_bumped_for_lyrics = false;
    let duration = if lyrics.is_some() {
        let requested = duration.unwrap_or(10);
        if requested < MIN_DURATION_WITH_LYRICS {
            duration_bumped_for_lyrics = true;
            Some(MIN_DURATION_WITH_LYRICS)
        } else {
            Some(requested)
        }
    } else {
        duration
    };

    // Send only the style portion to the enhancer so lyrics don't get re-absorbed.
    let prompt_for_enhancer = if lyrics.is_some() { &style_prompt } else { prompt };
    let enhanced_prompt = enhance_music_prompt(prompt_for_enhancer).await;

    let hera = hera_execution_agent();
    match hera.generate_music(&enhanced_prompt, duration, lyrics).await {
        Ok(res) => {
            let audio_url = res
                .get("audio_url")
                .and_then(|u| u.as_str())
                .unwrap_or("(no URL)");
            info!("🎵 [Hera] Music generated: {}", audio_url);

            let filename = audio_url.split('/').next_back().unwrap_or(audio_url);
            let public_url = format!("https://imaginos.ai/outputs/{}", filename);
            let actual_duration = res.get("duration").and_then(|d| d.as_u64());
            let bump_note = if duration_bumped_for_lyrics {
                " (duración ajustada a 30s para que la letra tenga espacio)"
            } else {
                ""
            };
            let response = match actual_duration {
                Some(secs) => format!(
                    "Music generated successfully! Enhanced prompt: \"{}\" ({}s){}\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the audio is delivered inline.",
                    enhanced_prompt, secs, bump_note, public_url
                ),
                None => format!(
                    "Music generated successfully! Enhanced prompt: \"{}\"{}\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the audio is delivered inline.",
                    enhanced_prompt, bump_note, public_url
                ),
            };

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: response,
            }
        }
        Err(e) => {
            tracing::error!("🎵 [Hera] Music generation failed: {:?}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Music generation failed: {}", e),
            }
        }
    }
}

pub(crate) async fn execute_animate_avatar(call: &ToolCall) -> ToolResult {
    let text = match call
        .arguments
        .get("text")
        .and_then(|t| t.as_str())
    {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: "Missing 'text' parameter — provide the text the avatar should say."
                    .to_string(),
            };
        }
    };
    let character = call
        .arguments
        .get("character")
        .and_then(|c| c.as_str())
        .unwrap_or("edu");
    let face_url = call
        .arguments
        .get("face_url")
        .and_then(|u| u.as_str());
    let voice = call
        .arguments
        .get("voice")
        .and_then(|v| v.as_str())
        .unwrap_or("paddi");

    let hera = hera_execution_agent();
    match hera.animate_avatar(text, character, face_url, Some(voice)).await {
        Ok(res) => {
            let video_url = res
                .get("video_url")
                .and_then(|u| u.as_str())
                .unwrap_or("(no URL)");
            info!("🎬 [Hera] Avatar animation generated: {}", video_url);

            let filename = video_url.split('/').next_back().unwrap_or(video_url);
            let public_url = format!("https://imaginos.ai/outputs/{}", filename);
            let response = format!(
                "Avatar animation generated successfully!\nMEDIA: {}\nInclude this MEDIA line EXACTLY as-is in your reply so the video is delivered inline.",
                public_url
            );

            ToolResult {
                name: call.name.clone(),
                success: true,
                output: response,
            }
        }
        Err(e) => {
            tracing::error!("🎬 [Hera] Avatar animation failed: {:?}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Avatar animation failed: {}", e),
            }
        }
    }
}

pub(crate) async fn execute_speak(call: &ToolCall) -> ToolResult {
    let text = call
        .arguments
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    let voice = call.arguments.get("voice").and_then(|v| v.as_str());

    let hera = hera_execution_agent();
    match hera.synthesize_speech(text, voice).await {
        Ok(result) => {
            info!("🔊 [Hera] Speech synthesized");
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Speech generated successfully: {}",
                    serde_json::to_string(&result).unwrap_or_default()
                ),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("TTS failed: {}", e),
        },
    }
}

pub(crate) async fn execute_video(call: &ToolCall) -> ToolResult {
    let prompt = call
        .arguments
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("A smooth cinematic video");

    let hera = hera_execution_agent();
    match hera.synthesize_video(prompt).await {
        Ok(result) => {
            info!("🎬 [Hera] Video generated");
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Video generated successfully: {}",
                    serde_json::to_string(&result).unwrap_or_default()
                ),
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Video generation failed: {}", e),
        },
    }
}

/// Sovereign image understanding — sends an image to the local VLM (Qwen2.5-VL @
/// HERA_VISION_URL, default :8083) and returns its answer. Used to describe or QA
/// images (e.g. detect headless people / deformed hands in generated covers).
/// `image` may be an http(s) URL or a local file path. `question` is optional.
pub(crate) async fn execute_review_image(call: &ToolCall) -> ToolResult {
    use base64::Engine as _;

    let fail = |msg: String| ToolResult {
        name: call.name.clone(),
        success: false,
        output: msg,
    };

    let image = call
        .arguments
        .get("image")
        .or_else(|| call.arguments.get("image_url"))
        .or_else(|| call.arguments.get("path"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if image.is_empty() {
        return fail("Missing 'image' (an http(s) URL or a local file path).".to_string());
    }
    let question = call
        .arguments
        .get("question")
        .or_else(|| call.arguments.get("prompt"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(
            "Describe this image briefly. Then, if it has obvious generation defects \
             (a person with a missing/cut-off head or face, deformed or extra hands/limbs, \
             melted faces, garbled text), add a final line 'DEFECT: <what>'. Otherwise add 'OK'.",
        );

    // Load the image bytes (remote URL or local path) and build a data URL.
    let bytes: Vec<u8> = if image.starts_with("http://") || image.starts_with("https://") {
        match reqwest::Client::new().get(&image).send().await {
            Ok(r) => match r.bytes().await {
                Ok(b) => b.to_vec(),
                Err(e) => return fail(format!("could not read image body: {e}")),
            },
            Err(e) => return fail(format!("could not fetch image: {e}")),
        }
    } else {
        match tokio::fs::read(&image).await {
            Ok(b) => b,
            Err(e) => return fail(format!("could not read file '{image}': {e}")),
        }
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let data_url = format!("data:image/png;base64,{b64}");

    let vision_url = std::env::var("HERA_VISION_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8083/v1/chat/completions".to_string());
    let payload = serde_json::json!({
        "model": "vision",
        "max_tokens": 200,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question}
            ]
        }]
    });

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
    {
        Ok(c) => c,
        Err(e) => return fail(format!("client build failed: {e}")),
    };
    match client.post(&vision_url).json(&payload).send().await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(j) => {
                let text = j["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if text.is_empty() {
                    return fail("vision model returned an empty response".to_string());
                }
                ToolResult {
                    name: call.name.clone(),
                    success: true,
                    output: text,
                }
            }
            Err(e) => fail(format!("could not parse vision response: {e}")),
        },
        Err(e) => fail(format!("vision request failed (is vision-review up @ {vision_url}?): {e}")),
    }
}
