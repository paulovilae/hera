use anyhow::{Result, anyhow};

pub async fn dispatch(action: &str, payload: serde_json::Value) -> Result<serde_json::Value> {
    let hera = hera_execution::agents::hera::Hera::new("http://127.0.0.1:3000");
    match action {
        "draw_image" => {
            let prompt = payload
                .get("prompt")
                .and_then(|value| value.as_str())
                .unwrap_or("A beautiful digital artwork");
            let width = payload
                .get("width")
                .and_then(|value| value.as_u64())
                .map(|value| value as u32);
            let height = payload
                .get("height")
                .and_then(|value| value.as_u64())
                .map(|value| value as u32);
            let result = hera
                .generate_image(prompt, None, width, height, None, None, None, None, None, None, None)
                .await?;
            Ok(result)
        }
        "speak_text" => {
            let text = payload
                .get("text")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("Missing `text`"))?;
            let voice = payload
                .get("voice")
                .and_then(|value| value.as_str());
            let result = hera.synthesize_speech(text, voice).await?;
            Ok(result)
        }
        "generate_video" => {
            let prompt = payload
                .get("prompt")
                .and_then(|value| value.as_str())
                .unwrap_or("A smooth cinematic video");
            let result = hera.synthesize_video(prompt).await?;
            Ok(result)
        }
        _ => Err(anyhow!("Unsupported media action `{}`", action)),
    }
}
