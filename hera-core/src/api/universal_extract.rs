use axum::{
    extract::{Multipart, State},
    response::sse::{Event, Sse},
};
use futures_util::stream::Stream;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use hera_execution::tools::native_xls::execute_native_xls;
use hera_execution::tools::native_ocr::execute_native_ocr;
use serde_json::json;

use crate::api::routes::ApiState;

pub async fn hera_extract_sse(
    State(state): State<Arc<ApiState>>,
    mut multipart: Multipart,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel(100);

    // Read multipart fields before spawning
    let mut file_bytes = Vec::new();
    let mut file_name_opt = None;
    let mut custom_prompt = "Extrae todos los datos relevantes de este documento y estructúralos en formato JSON exacto sin texto adicional.".to_string();
    let mut p_schema: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" || name == "documentBase64" {
            if let Some(fn_name) = field.file_name() {
                file_name_opt = Some(fn_name.to_lowercase());
            }
            if let Ok(data) = field.bytes().await {
                let data_str = String::from_utf8_lossy(&data).to_string();
                if data_str.starts_with("data:") && data_str.contains("base64,") {
                    if let Some(b64) = data_str.split("base64,").nth(1) {
                        if let Ok(decoded) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim()) {
                            file_bytes = decoded;
                            let mime = data_str.split("data:").nth(1).unwrap_or("").split(";").next().unwrap_or("");
                            if mime.contains("pdf") { file_name_opt = Some("doc.pdf".to_string()); }
                            else if mime.contains("excel") || mime.contains("spreadsheet") { file_name_opt = Some("doc.xlsx".to_string()); }
                            else if mime.contains("image") { file_name_opt = Some("doc.png".to_string()); }
                        }
                    }
                } else {
                    file_bytes = data.to_vec();
                }
            }
        } else if name == "schema" {
            if let Ok(text) = field.text().await {
                if !text.trim().is_empty() { p_schema = Some(text); }
            }
        }
    }

    let file_name = file_name_opt.unwrap_or_else(|| "unknown.bin".to_string());

    tokio::spawn(async move {
        let t0 = Instant::now();

        // ─── Dynamic Schema Engine ───
        #[derive(serde::Deserialize, Clone)]
        struct SchemaField {
            key: String,
            description: String,
            #[serde(default = "default_type_string")]
            #[serde(rename = "type")]
            field_type: String, // "string" or "number"
        }
        fn default_type_string() -> String { "string".to_string() }

        let schema_fields: Vec<SchemaField> = p_schema
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_else(|| vec![
                SchemaField { key: "name".into(), description: "Nombre principal".into(), field_type: "string".into() },
                SchemaField { key: "value".into(), description: "Valor o precio".into(), field_type: "number".into() },
            ]);

        // Helper to clean prices/numbers
        fn parse_number(raw: &str) -> serde_json::Value {
            let cleaned: String = raw.chars().filter(|c| c.is_ascii_digit() || *c == '.' || *c == ',').collect();
            let dot_count = cleaned.matches('.').count();
            let final_str = if dot_count > 1 || (dot_count == 1 && cleaned.len() - cleaned.rfind('.').unwrap_or(0) == 4) {
                cleaned.replace('.', "").replace(',', "")
            } else {
                cleaned.replace(',', "")
            };
            if let Ok(n) = final_str.parse::<f64>() { serde_json::json!(n as i64) } else { serde_json::json!("") }
        }

        // ─── LLM Header Mapper (Dynamic) ───
        async fn llm_map_headers(
            engine: &std::sync::Arc<dyn crate::ai::LLMEngine + Send + Sync>,
            headers: &[String],
            fields: &[SchemaField],
        ) -> Option<serde_json::Value> {
            let header_str = headers.iter().enumerate().map(|(i, h)| format!("{}: \"{}\"", i, h)).collect::<Vec<_>>().join(", ");
            
            let mut sys = String::from("Eres un mapeador de columnas. Recibes una lista de encabezados y mapeas las columnas requeridas.\n\nReglas:\n");
            for f in fields {
                sys.push_str(&format!("- {}: {}\n", f.key, f.description));
            }
            
            let exact_json_format = fields.iter().map(|f| format!("\"{}\":IDX", f.key)).collect::<Vec<_>>().join(",");
            sys.push_str(&format!("\nResponde SOLO un JSON con este formato exacto (sin ```json):\n{{{}}}\nSi una columna no existe, usa -1.", exact_json_format));

            let user_msg = format!("Encabezados: {}", header_str);

            let req = crate::ai::ChatRequest {
                model: "local".to_string(), vision_model: None, tts_model: None, stt_model: None,
                messages: vec![
                    crate::ai::ChatMessage { role: "system".to_string(), content: crate::ai::MessageContent::Text(sys) },
                    crate::ai::ChatMessage { role: "user".to_string(), content: crate::ai::MessageContent::Text(user_msg) }
                ],
                temperature: Some(0.0), max_tokens: Some(150),
                top_p: None, top_k: None, presence_penalty: None, frequency_penalty: None, repeat_penalty: None, seed: None, stop: None, endpoint: None,
                api_key: None, provider: Some("local".to_string()), stream: Some(true), nsfw: None, tools: None, tool_choice: None, reasoning_effort: None,
            };

            match engine.generate_stream(req).await {
                Ok(mut stream) => {
                    let mut acc = String::new();
                    while let Some(Ok(c)) = stream.recv().await {
                        if let Some(txt) = c.choices.first().and_then(|x| x.delta.content.as_ref()) { acc.push_str(txt); }
                    }
                    let mut raw = acc.trim().to_string();
                    if let Some(e) = raw.find("</think>") { raw = raw[e + 8..].trim().to_string(); }
                    if raw.starts_with("```json") { raw = raw.trim_start_matches("```json").trim_end_matches("```").to_string(); }
                    else if raw.starts_with("```") { raw = raw.trim_start_matches("```").trim_end_matches("```").to_string(); }
                    tracing::info!("🧠 [Dynamic Schema] Mapped: {}", raw.trim());
                    serde_json::from_str::<serde_json::Value>(raw.trim()).ok()
                }
                Err(e) => { tracing::error!("❌ LLM Error: {}", e); None }
            }
        }

        // ─── Row mapper (Dynamic) ───
        fn map_row_by_indices(cols: &[&str], mapping: &serde_json::Value, fields: &[SchemaField]) -> Option<serde_json::Value> {
            let mut row_obj = serde_json::Map::new();
            let mut has_meaningful_string = false;
            let mut has_meaningful_number = false;
            let mut schema_requires_numbers = false;

            for field in fields {
                let idx = mapping.get(&field.key).and_then(|v| v.as_i64()).and_then(|i| if i >= 0 { Some(i as usize) } else { None });
                let raw_val = idx.and_then(|i| cols.get(i)).map(|v| v.trim()).unwrap_or("");
                
                if field.field_type == "number" {
                    schema_requires_numbers = true;
                    let num_val = parse_number(raw_val);
                    if num_val.is_number() { has_meaningful_number = true; }
                    row_obj.insert(field.key.clone(), num_val);
                } else {
                    if !raw_val.is_empty() && raw_val != "-" { has_meaningful_string = true; }
                    row_obj.insert(field.key.clone(), serde_json::json!(raw_val));
                }
            }

            // UNIVERSAL FILTERING RULE:
            // If the client's schema expects numeric data (like prices), a row consisting ONLY of text 
            // (e.g. footers, "Contact Us on Whatsapp", "Cardiology Category") is invalid.
            let is_valid = if schema_requires_numbers {
                has_meaningful_string && has_meaningful_number
            } else {
                has_meaningful_string
            };

            if is_valid { Some(serde_json::Value::Object(row_obj)) } else { None }
        }

        macro_rules! send_event {
            ($tx:expr, $step:expr, $status:expr, $data:expr) => {{
                let mut data_opt: Option<serde_json::Value> = $data;
                let mut payload = json!({
                    "step": $step,
                    "status": $status
                });
                if let Some(d) = data_opt {
                    payload.as_object_mut().unwrap().insert("data".to_string(), d);
                }
                let event = Event::default().data(payload.to_string());
                let _ = $tx.send(Ok(event)).await;
            }};
        }

        if file_bytes.is_empty() {
            send_event!(tx, "error", "No se encontró ningún archivo válido en la petición.", None);
            return;
        }

        send_event!(tx, "classifying", &format!("Archivo recibido ({}). Analizando formato...", file_name), None);
        
        let is_excel = file_name.ends_with(".xls") || file_name.ends_with(".xlsx") || file_name.ends_with(".ods");
        let is_pdf_or_img = file_name.ends_with(".pdf") || file_name.ends_with(".png") || file_name.ends_with(".jpg") || file_name.ends_with(".jpeg");

        // ═══════════════════════════════════════════════════════════════
        // EXCEL PATH: Calamine parse (instant) → LLM header map (~1s) → Rust row mapping (instant)
        // ═══════════════════════════════════════════════════════════════
        if is_excel {
            send_event!(tx, "routing", "Ruta: Excel → Calamine + LLM Header Intelligence", None);
            let raw_data = match execute_native_xls("temp_obj", &file_bytes).await {
                Ok(data) => data,
                Err(e) => {
                    send_event!(tx, "error", &format!("Fallo en extracción Excel: {}", e), None);
                    return;
                }
            };

            tracing::info!("📊 [Excel] Raw calamine output (first 500 chars): {}", &raw_data[..raw_data.len().min(500)]);

            let lines: Vec<&str> = raw_data.lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with("[Native") && !l.starts_with("\\n##") && !l.contains("## Sheet:"))
                .collect();

            if lines.is_empty() {
                send_event!(tx, "error", "El archivo Excel no contiene filas de datos.", None);
                return;
            }

            // Find the header row: first row where at least 2 cells have non-empty content
            let mut header_row_idx = 0;
            for (i, line) in lines.iter().enumerate() {
                let non_empty_count = line.split('|').filter(|c| !c.trim().is_empty()).count();
                if non_empty_count >= 2 {
                    header_row_idx = i;
                    break;
                }
            }

            let headers: Vec<String> = lines[header_row_idx].split('|').map(|h| h.trim().to_lowercase()).collect();
            tracing::info!("📊 [Excel] Header row {} → {:?}", header_row_idx, headers);

            send_event!(tx, "structuring", "Datos extraídos. LLM mapeando encabezados...", None);

            let engine = state.local_engine.clone();
            let mapping = llm_map_headers(&engine, &headers, &schema_fields).await;

            if let Some(ref col_map) = mapping {
                tracing::info!("🧠 [Excel] LLM column mapping: {}", col_map);

                let mut services: Vec<serde_json::Value> = Vec::new();
                let name_idx = col_map.get("name").and_then(|v| v.as_i64()).and_then(|i| if i >= 0 { Some(i as usize) } else { None }).unwrap_or(0);

                for line in &lines[(header_row_idx + 1)..] {
                    let cols: Vec<&str> = line.split('|').map(|c| c.trim()).collect();
                    let name_val = cols.get(name_idx).unwrap_or(&"").trim();
                    if let Some(valid_row) = map_row_by_indices(&cols, col_map, &schema_fields) {
                        services.push(valid_row);
                    }
                }

                let elapsed_ms = t0.elapsed().as_millis();
                tracing::info!("✅ [Excel] {} rows mapped in {}ms (LLM headers + Rust rows)", services.len(), elapsed_ms);
                let result = serde_json::json!(services);
                let _ = send_event!(tx, "finished", &format!("✅ {} servicios en {}ms (LLM headers + Rust data)", services.len(), elapsed_ms), Some(result));
            } else {
                send_event!(tx, "error", "LLM no pudo mapear los encabezados del Excel.", None);
            }
            return;
        }

        // ═══════════════════════════════════════════════════════════════
        // IMAGE/PDF PATH: OCR → LLM (full structuring for unstructured text)
        // ═══════════════════════════════════════════════════════════════
        if is_pdf_or_img {
            send_event!(tx, "routing", "Ruta: OCR Nativo + LLM Inteligencia", None);
            let raw_data = match execute_native_ocr("temp_obj", true, &file_bytes).await {
                Ok(data) => data,
                Err(e) => {
                    send_event!(tx, "error", &format!("Fallo en OCR: {}", e), None);
                    return;
                }
            };

            // Check if OCR produced tabular text (has pipes/tabs → can use header mapping)
            let has_separator = raw_data.contains('|') || raw_data.contains('\t');
            let separator = if raw_data.contains('|') { '|' } else { '\t' };
            let lines: Vec<&str> = raw_data.lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with("[Native"))
                .collect();

            if has_separator && lines.len() >= 2 {
                // Tabular OCR → same hybrid approach as Excel
                let headers: Vec<String> = lines[0].split(separator).map(|h| h.trim().to_lowercase()).collect();

                if headers.len() >= 2 {
                    send_event!(tx, "structuring", "OCR tabular detectado. LLM mapeando encabezados...", None);
                    let engine = state.local_engine.clone();
                    let mapping = llm_map_headers(&engine, &headers, &schema_fields).await;

                    if let Some(ref col_map) = mapping {
                        let name_idx = col_map.get("name").and_then(|v| v.as_i64()).and_then(|i| if i >= 0 { Some(i as usize) } else { None }).unwrap_or(0);
                        let mut services: Vec<serde_json::Value> = Vec::new();
                        for line in &lines[1..] {
                            let cols: Vec<&str> = line.split(separator).map(|c| c.trim()).collect();
                            let name_val = cols.get(name_idx).unwrap_or(&"").trim();
                            if let Some(valid_row) = map_row_by_indices(&cols, col_map, &schema_fields) {
                                services.push(valid_row);
                            }
                        }

                        if !services.is_empty() {
                            let elapsed_ms = t0.elapsed().as_millis();
                            let result = serde_json::json!(services);
                            let _ = send_event!(tx, "finished", &format!("✅ OCR: {} servicios en {}ms (LLM headers + Rust data)", services.len(), elapsed_ms), Some(result));
                            return;
                        }
                    }
                }
            }

            // Non-tabular OCR → full LLM structuring (unstructured text)
            send_event!(tx, "structuring", "OCR no tabular. LLM estructurando texto completo...", None);
            let engine = state.local_engine.clone();
            
            let mut sys_prompt = String::from("Eres un motor de extracción AI. Extrae la info del texto OCR y responde SOLO con un arreglo JSON válido: [{...}]. No uses ```json ni Markdown.\nFormato esperado por objeto:\n{");
            let json_keys = schema_fields.iter().map(|f| format!("\"{}\": \"{}\"", f.key, f.field_type)).collect::<Vec<_>>().join(", ");
            sys_prompt.push_str(&json_keys);
            sys_prompt.push_str("}\nLimpia los números si es type: number (ej: $ 15.000 → 15000). Si un campo no existe, usa un string vacío o 0.");
            
            let combined_prompt = format!("Instrucción: {}\n\nData Cruda:\n{}", custom_prompt, raw_data);

            let req = crate::ai::ChatRequest {
                model: "local".to_string(),
                vision_model: None, tts_model: None, stt_model: None,
                messages: vec![
                    crate::ai::ChatMessage { role: "system".to_string(), content: crate::ai::MessageContent::Text(sys_prompt.to_string()) },
                    crate::ai::ChatMessage { role: "user".to_string(), content: crate::ai::MessageContent::Text(combined_prompt) }
                ],
                temperature: Some(0.1), max_tokens: Some(4000),
                top_p: None, top_k: None, presence_penalty: None, frequency_penalty: None, repeat_penalty: None, seed: None, stop: None, endpoint: None,
                api_key: None, provider: Some("local".to_string()), stream: Some(true),
                nsfw: None, tools: None, tool_choice: None, reasoning_effort: None,
            };

            match engine.generate_stream(req).await {
                Ok(mut rx_stream) => {
                    let mut accumulated = String::new();
                    let mut chunk_count = 0;
                    while let Some(res_chunk) = rx_stream.recv().await {
                        match res_chunk {
                            Ok(chunk) => {
                                if let Some(content) = chunk.choices.first().and_then(|c| c.delta.content.clone()) {
                                    accumulated.push_str(&content);
                                    chunk_count += 1;
                                    if chunk_count % 15 == 0 {
                                        let _ = send_event!(tx, "structuring", "Hera LLM procesando OCR (manteniendo conexión)...", None);
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = send_event!(tx, "error", &format!("Fallo en stream Hera LLM: {}", e), None);
                                return;
                            }
                        }
                    }
                    let mut raw_str = accumulated.trim();
                    if raw_str.starts_with("```json") { raw_str = raw_str.trim_start_matches("```json").trim_end_matches("```"); }
                    else if raw_str.starts_with("```") { raw_str = raw_str.trim_start_matches("```").trim_end_matches("```"); }
                    let raw_str = raw_str.trim();
                    match serde_json::from_str::<serde_json::Value>(raw_str) {
                        Ok(j) => {
                            let elapsed_ms = t0.elapsed().as_millis();
                            let _ = send_event!(tx, "finished", &format!("✅ Estructuración OCR completa en {}ms.", elapsed_ms), Some(j));
                        }
                        Err(_) => {
                            let fallback = json!({"raw_text": raw_str});
                            let _ = send_event!(tx, "finished", "Estructuración completada. Revisa el texto crudo.", Some(fallback));
                        }
                    }
                }
                Err(e) => {
                    let _ = send_event!(tx, "error", &format!("Fallo en Hera LLM: {}", e), None);
                }
            }
            return;
        }

        // Unsupported format
        send_event!(tx, "error", "Formato de archivo no soportado. Por favor sube Excel, PDF o Imagen.", None);
        tracing::info!("🏁 [Universal Extract] Tokio thread finishing.");
    });

    axum::response::sse::Sse::new(ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(10))
            .text("keep-alive"))
}
