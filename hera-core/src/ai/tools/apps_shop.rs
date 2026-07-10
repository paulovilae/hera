//! Shop app tool executors — tienda multi-tenant sobre `os-admin-kit`.
//!
//! Clientes DELGADOS del engine `shop` de `os-admin-kit`: la lógica de negocio
//! (productos, inventario, órdenes, carritos abandonados) vive en el app
//! (`Apps/OS-v3/engine-kit`), no se duplica acá — mismo contrato que
//! `apps_construvendo`/`apps_vetra`. Los dos últimos executors (descripción de
//! producto, SEO meta) son Tier-0 puro: generan texto localmente vía la
//! conexión IPC directa a Hera, sin tocar el engine.
use crate::ai::tool_executor::{ToolCall, ToolResult};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Base del app Shop (`os-admin-kit`, engine `shop`). Override con `STORE_URL`.
fn base_url() -> String {
    std::env::var("STORE_URL").unwrap_or_else(|_| "http://127.0.0.1:5203".to_string())
}

const HERA_SOCKET: &str = "/tmp/hera-core.sock";

/// Pretty-print de un `View` JSON devuelto por el engine para el output del tool.
fn pretty(json: &Value) -> String {
    serde_json::to_string_pretty(json).unwrap_or_else(|_| json.to_string())
}

/// POST delgado a `{STORE_URL}/api/engines/shop/action/{action}`. La lógica y
/// los datos de la tienda viven en el engine, acá solo se transporta el JSON.
async fn call_shop_action(action: &str, body: Value) -> Result<Value, String> {
    let url = format!("{}/api/engines/shop/action/{}", base_url(), action);
    let client = reqwest::Client::new();
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => resp
            .json::<Value>()
            .await
            .map_err(|e| format!("No pude leer la respuesta de la tienda: {e}")),
        Ok(resp) => Err(format!("La tienda respondió con error: {}", resp.status())),
        Err(e) => Err(format!("No pude consultar la tienda ahora mismo: {e}")),
    }
}

/// `list_products` → engine action `list_products`.
pub(crate) async fn execute_list_products(call: &ToolCall) -> ToolResult {
    let tenant_id = call
        .arguments
        .get("tenant_id")
        .and_then(|v| v.as_str())
        .unwrap_or("imaginos-store");
    let body = json!({ "tenant_id": tenant_id });

    match call_shop_action("list_products", body).await {
        Ok(view) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: pretty(&view),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e,
        },
    }
}

/// `create_product` → engine action `create_product`. Requiere `title`.
pub(crate) async fn execute_create_product(call: &ToolCall) -> ToolResult {
    let title = call
        .arguments
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if title.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Falta el título del producto (parámetro title).".into(),
        };
    }

    let mut body = serde_json::Map::new();
    body.insert("title".into(), json!(title));
    for key in [
        "price_cents",
        "slug",
        "description",
        "currency",
        "status",
        "tenant_id",
    ] {
        if let Some(v) = call.arguments.get(key) {
            body.insert(key.into(), v.clone());
        }
    }

    match call_shop_action("create_product", Value::Object(body)).await {
        Ok(view) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: pretty(&view),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e,
        },
    }
}

/// `update_inventory` → engine action `update_inventory`. Requiere `sku` o
/// `variant_id`, más `available`.
pub(crate) async fn execute_update_inventory(call: &ToolCall) -> ToolResult {
    let sku = call
        .arguments
        .get("sku")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty());
    let variant_id = call
        .arguments
        .get("variant_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty());
    if sku.is_none() && variant_id.is_none() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Necesito sku o variant_id para actualizar el inventario.".into(),
        };
    }
    let Some(available) = call.arguments.get("available").and_then(|v| v.as_i64()) else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Necesito la cantidad disponible (available) para actualizar el inventario."
                .into(),
        };
    };

    let mut body = serde_json::Map::new();
    if let Some(sku) = sku {
        body.insert("sku".into(), json!(sku));
    }
    if let Some(variant_id) = variant_id {
        body.insert("variant_id".into(), json!(variant_id));
    }
    body.insert("available".into(), json!(available));
    if let Some(v) = call.arguments.get("tenant_id") {
        body.insert("tenant_id".into(), v.clone());
    }

    match call_shop_action("update_inventory", Value::Object(body)).await {
        Ok(view) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: pretty(&view),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e,
        },
    }
}

/// `shop_create_order` → engine action `create_order`. Requiere `items` o
/// `cart_id`.
pub(crate) async fn execute_shop_create_order(call: &ToolCall) -> ToolResult {
    let items = call.arguments.get("items").filter(|v| v.is_array());
    let cart_id = call
        .arguments
        .get("cart_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty());
    if items.is_none() && cart_id.is_none() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Necesito items o cart_id para crear la orden.".into(),
        };
    }

    let mut body = serde_json::Map::new();
    if let Some(items) = items {
        body.insert("items".into(), items.clone());
    }
    if let Some(cart_id) = cart_id {
        body.insert("cart_id".into(), json!(cart_id));
    }
    if let Some(v) = call.arguments.get("user_email") {
        body.insert("user_email".into(), v.clone());
    }
    if let Some(v) = call.arguments.get("tenant_id") {
        body.insert("tenant_id".into(), v.clone());
    }

    match call_shop_action("create_order", Value::Object(body)).await {
        Ok(view) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: pretty(&view),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e,
        },
    }
}

/// `shop_check_inventory` → engine action `check_inventory`. Requiere `sku` o
/// `variant_id`.
pub(crate) async fn execute_shop_check_inventory(call: &ToolCall) -> ToolResult {
    let sku = call
        .arguments
        .get("sku")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty());
    let variant_id = call
        .arguments
        .get("variant_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty());
    if sku.is_none() && variant_id.is_none() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Necesito sku o variant_id para consultar inventario.".into(),
        };
    }

    let mut body = serde_json::Map::new();
    if let Some(sku) = sku {
        body.insert("sku".into(), json!(sku));
    }
    if let Some(variant_id) = variant_id {
        body.insert("variant_id".into(), json!(variant_id));
    }
    if let Some(v) = call.arguments.get("tenant_id") {
        body.insert("tenant_id".into(), v.clone());
    }

    match call_shop_action("check_inventory", Value::Object(body)).await {
        Ok(view) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: pretty(&view),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e,
        },
    }
}

/// `abandoned_cart_to_lead` → engine action `abandoned_cart_to_lead`. Requiere
/// `cart_id`.
pub(crate) async fn execute_abandoned_cart_to_lead(call: &ToolCall) -> ToolResult {
    let cart_id = call
        .arguments
        .get("cart_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if cart_id.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Falta el cart_id del carrito abandonado.".into(),
        };
    }

    let mut body = serde_json::Map::new();
    body.insert("cart_id".into(), json!(cart_id));
    if let Some(v) = call.arguments.get("email") {
        body.insert("email".into(), v.clone());
    }
    if let Some(v) = call.arguments.get("tenant_id") {
        body.insert("tenant_id".into(), v.clone());
    }

    match call_shop_action("abandoned_cart_to_lead", Value::Object(body)).await {
        Ok(view) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: pretty(&view),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e,
        },
    }
}

// ── Tier-0 puro: generación local de texto, sin tocar el engine ────────────
//
// Conexión directa al socket IPC de Hera (mismo patrón que
// `platform::execute_corporate_research`'s synthesis step — no hay un helper
// pub(crate) reusable ahí porque `parse_ipc_result` es privado a ese módulo y
// esta unidad no toca `platform.rs`, así que se replica localmente la
// secuencia connect → write → shutdown → read → parse).

/// Manda un turno `system` + `user` a Hera vía IPC (`/tmp/hera-core.sock`,
/// acción `generate`) y devuelve el texto generado.
async fn generate_via_hera(
    system_prompt: String,
    user_prompt: String,
    max_tokens: u32,
) -> Result<String, String> {
    let mut stream = UnixStream::connect(HERA_SOCKET)
        .await
        .map_err(|e| format!("No pude conectar con Hera IPC: {e}"))?;

    let ipc_request = json!({
        "action": "generate",
        "payload": {
            "app": "shop",
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "temperature": 0.4,
            "max_tokens": max_tokens,
            "permissions": []
        }
    });

    let payload = format!("{}\n", ipc_request);
    stream
        .write_all(payload.as_bytes())
        .await
        .map_err(|e| format!("Error escribiendo al socket de Hera: {e}"))?;
    stream
        .shutdown()
        .await
        .map_err(|e| format!("Error cerrando el socket de Hera: {e}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .map_err(|e| format!("Error leyendo la respuesta de Hera: {e}"))?;

    parse_ipc_result(&response)
}

/// Extrae el texto final de un stream de mensajes IPC de Hera (línea por
/// línea, `status: success|chunk|error`). Copia local de
/// `platform::parse_ipc_result` (privada a ese módulo).
fn parse_ipc_result(response: &str) -> Result<String, String> {
    let mut accumulated_text = String::new();
    let mut final_result = None;

    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let Ok(message) = serde_json::from_str::<Value>(line) else {
            continue;
        };

        match message.get("status").and_then(|value| value.as_str()) {
            Some("success") => {
                if let Some(result) = message
                    .pointer("/data/result")
                    .and_then(|value| value.as_str())
                {
                    final_result = Some(result.to_string());
                }
            }
            Some("chunk") => {
                if let Some(text) = message
                    .pointer("/data/text")
                    .and_then(|value| value.as_str())
                {
                    accumulated_text.push_str(text);
                }
            }
            Some("error") => {
                let error = message
                    .pointer("/data/error")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown Hera IPC error");
                return Err(error.to_string());
            }
            _ => {}
        }
    }

    if let Some(result) = final_result {
        Ok(result)
    } else if !accumulated_text.is_empty() {
        Ok(accumulated_text)
    } else {
        Err("No content in Hera IPC response".to_string())
    }
}

/// `generate_product_description` — Tier-0: redacta una descripción corta de
/// producto a partir de `title` (requerido), `features?`, `tone?`, `lang?`
/// (default "es").
pub(crate) async fn execute_generate_product_description(call: &ToolCall) -> ToolResult {
    let title = call
        .arguments
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if title.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Falta el título del producto (parámetro title).".into(),
        };
    }
    let features = call
        .arguments
        .get("features")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let tone = call
        .arguments
        .get("tone")
        .and_then(|v| v.as_str())
        .unwrap_or("cercano y profesional")
        .trim();
    let lang = call
        .arguments
        .get("lang")
        .and_then(|v| v.as_str())
        .unwrap_or("es")
        .trim();

    let (system_prompt, user_prompt) = if lang == "en" {
        (
            "You are an e-commerce copywriter. Write a short, persuasive product description \
             (2-4 sentences) in English. Do not invent features you were not given. Return plain \
             text only — no markdown, no quotes."
                .to_string(),
            format!(
                "Product title: {title}\nKey features: {}\nTone: {tone}",
                if features.is_empty() { "(none given)" } else { features }
            ),
        )
    } else {
        (
            "Eres un redactor de e-commerce. Escribe una descripción de producto corta y \
             persuasiva (2-4 oraciones) en español. No inventes características que no te \
             dieron. Devuelve solo texto plano, sin markdown ni comillas."
                .to_string(),
            format!(
                "Título del producto: {title}\nCaracterísticas clave: {}\nTono: {tone}",
                if features.is_empty() { "(no especificadas)" } else { features }
            ),
        )
    };

    match generate_via_hera(system_prompt, user_prompt, 300).await {
        Ok(text) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: text.trim().to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("No pude generar la descripción: {e}"),
        },
    }
}

/// `suggest_seo_meta` — Tier-0: sugiere `seo_title` (<=60 chars) y
/// `seo_description` (<=155 chars) a partir de `title` (requerido),
/// `description?`, `lang?` (default "es").
pub(crate) async fn execute_suggest_seo_meta(call: &ToolCall) -> ToolResult {
    let title = call
        .arguments
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if title.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Falta el título del producto (parámetro title).".into(),
        };
    }
    let description = call
        .arguments
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    let lang = call
        .arguments
        .get("lang")
        .and_then(|v| v.as_str())
        .unwrap_or("es")
        .trim();

    let (system_prompt, user_prompt) = if lang == "en" {
        (
            "You are an SEO copywriter. Given a product title and description, produce a JSON \
             object with exactly two keys: seo_title (<=60 chars) and seo_description \
             (<=155 chars). Do not invent facts. Return ONLY the JSON object, no markdown fences."
                .to_string(),
            format!(
                "Title: {title}\nDescription: {}",
                if description.is_empty() { "(none given)" } else { description }
            ),
        )
    } else {
        (
            "Eres un redactor SEO. Dado un título y una descripción de producto, genera un \
             objeto JSON con exactamente dos claves: seo_title (<=60 caracteres) y \
             seo_description (<=155 caracteres). No inventes datos. Devuelve SOLO el objeto \
             JSON, sin fences de markdown."
                .to_string(),
            format!(
                "Título: {title}\nDescripción: {}",
                if description.is_empty() { "(no especificada)" } else { description }
            ),
        )
    };

    match generate_via_hera(system_prompt, user_prompt, 200).await {
        Ok(text) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: text.trim().to_string(),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("No pude generar el SEO meta: {e}"),
        },
    }
}
