//! Movilo app tool executors
use crate::ai::tool_executor::{ToolCall, ToolResult};
use crate::ai::tools::data::execute_memento_query;

// Postgres `translate()` folds accents on the column side; both sides also get
// `lower()` to be case-insensitive. Spanish dataset has rows like "Odontólogos"
// while users type "odontologos" — without folding the ILIKE never matched.
const ACCENT_FROM: &str = "ÁÀÄÂÃÉÈËÊÍÌÏÎÓÒÖÔÕÚÙÜÛÑÇáàäâãéèëêíìïîóòöôõúùüûñç";
const ACCENT_TO: &str = "AAAAAEEEEIIIIOOOOOUUUUNCaaaaaeeeeiiiioooooouuuunc";

fn fold_accents_lower(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'á' | 'à' | 'ä' | 'â' | 'ã' | 'Á' | 'À' | 'Ä' | 'Â' | 'Ã' => 'a',
            'é' | 'è' | 'ë' | 'ê' | 'É' | 'È' | 'Ë' | 'Ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' | 'Í' | 'Ì' | 'Ï' | 'Î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' | 'õ' | 'Ó' | 'Ò' | 'Ö' | 'Ô' | 'Õ' => 'o',
            'ú' | 'ù' | 'ü' | 'û' | 'Ú' | 'Ù' | 'Ü' | 'Û' => 'u',
            'ñ' | 'Ñ' => 'n',
            'ç' | 'Ç' => 'c',
            other => other.to_ascii_lowercase(),
        })
        .collect()
}

// Strip a trailing Spanish plural suffix so a user typing "odontólogos" or
// "doctores" still matches singular rows like "Odontólogo" or "Doctor".
// Conservative: only strips if the remainder is ≥ 3 chars, so short words
// ("mas", "los") are left alone.
fn singularize_es(s: &str) -> &str {
    if s.len() > 4 && s.ends_with("es") {
        &s[..s.len() - 2]
    } else if s.len() > 3 && s.ends_with('s') {
        &s[..s.len() - 1]
    } else {
        s
    }
}

// Reduce to a common prefix stem so cross-nominal queries match. The LLM
// often passes the *field* form ("odontología") while the DB stores the
// *practitioner* form ("Odontólogo") and vice versa. Both share a 6-char
// prefix ("odonto") that survives any Spanish derivation. Keep at least 6
// chars when the input is long enough; leave shorter inputs untouched so
// we don't over-match short keywords ("cali", "ips").
fn stem_prefix_es(s: &str) -> &str {
    const STEM_LEN: usize = 6;
    if s.len() > STEM_LEN {
        // Walk char boundaries to avoid slicing through a UTF-8 sequence —
        // by this point the string is ASCII (post-fold), but defensive.
        match s.char_indices().nth(STEM_LEN) {
            Some((idx, _)) => &s[..idx],
            None => s,
        }
    } else {
        s
    }
}

/// Map a free-form specialty input to one of the widget's canonical tab IDs.
/// Tab IDs come from os-provider-map.js: Todos / Clínica / Especialista /
/// Odontólogo / Laboratorio / Farmacia / Veterinaria. Everything that doesn't
/// fit a category falls back to "Todos" so the user sees the full directory
/// (the chat text already lists the specific matches).
fn canonical_widget_category(raw: &str) -> &'static str {
    let folded = fold_accents_lower(raw);
    if folded.contains("odontolog") || folded.contains("dental") || folded.contains("dentist") {
        "Odontólogo"
    } else if folded.contains("farmac") || folded.contains("droguer") {
        "Farmacia"
    } else if folded.contains("veterinar") || folded.contains("mascot") {
        "Veterinaria"
    } else if folded.contains("laborator") {
        "Laboratorio"
    } else if folded.contains("clinic") || folded.contains("ips") || folded.contains("centro") {
        "Clínica"
    } else if folded.contains("especial") || folded.contains("medic") || !folded.trim().is_empty() {
        "Especialista"
    } else {
        "Todos"
    }
}

fn escape_attr(raw: &str) -> String {
    raw.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn folded_like(column: &str, raw_input: &str) -> String {
    let lowered = fold_accents_lower(raw_input);
    let singular = singularize_es(&lowered);
    let stemmed = stem_prefix_es(singular);
    let folded = stemmed.replace('\'', "''");
    format!("lower(translate({column}, '{ACCENT_FROM}', '{ACCENT_TO}')) LIKE '%{folded}%'")
}

/// Render provider rows as a short, user-facing Spanish list.
///
/// The tool output is consumed by a weak local model on a streaming route
/// (`movilo_widget` is `prefer_stream`), which frequently echoes the tool result
/// verbatim. So the output MUST already be presentable: no raw JSON, no SQL, no
/// internal `[[SYSTEM DIRECTIVE]]` framing — whatever the model echoes has to be
/// clean. Dedups by company so the LEFT JOIN on services doesn't repeat a
/// provider once per service row.
fn format_provider_list(rows: &[serde_json::Value]) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut lines = Vec::new();
    for row in rows {
        let name = row
            .get("company_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if name.is_empty() || !seen.insert(name.to_string()) {
            continue;
        }
        let ptype = row
            .get("provider_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let phone = row
            .get("phone")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let mut parts = vec![format!("**{name}**")];
        if !ptype.is_empty() {
            parts.push(ptype.to_string());
        }
        if !phone.is_empty() {
            parts.push(format!("Tel: {phone}"));
        }
        lines.push(format!("- {}", parts.join(" — ")));
    }
    lines.join("\n")
}

pub(crate) async fn execute_movilo_search_providers(call: &ToolCall) -> ToolResult {
    let city = call
        .arguments
        .get("city")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    // Accept several aliases — the LLM sometimes invents arg names ("service",
    // "type", "provider_type") that aren't in the schema. Without these aliases
    // the value silently falls through to no filter and the tool returns the
    // city's full directory, which confuses the LLM into hallucinated answers.
    let specialty = call
        .arguments
        .get("specialty")
        .or_else(|| call.arguments.get("provider_type"))
        .or_else(|| call.arguments.get("type"))
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let keyword = call
        .arguments
        .get("service_keyword")
        .or_else(|| call.arguments.get("service"))
        .or_else(|| call.arguments.get("keyword"))
        .and_then(|k| k.as_str())
        .unwrap_or("");

    let mut conditions = vec!["p.status = 'Aprobado'".to_string()];
    if !city.is_empty() {
        conditions.push(folded_like("p.city", city));
    }
    // El LLM a veces mete la especialidad en `service` y viceversa (p.ej.
    // "cardiologo" llegó como service → filtraba s.name y daba 0 aunque existe
    // el provider_type "Cardiología"). Si solo hay UN término, mátchealo contra
    // provider_type O nombre de servicio. Con AMBOS, filtra preciso (tipo+servicio).
    if !specialty.is_empty() && !keyword.is_empty() {
        conditions.push(folded_like("p.provider_type", specialty));
        conditions.push(folded_like("s.name", keyword));
    } else {
        let term = if !specialty.is_empty() { specialty } else { keyword };
        if !term.is_empty() {
            conditions.push(format!(
                "({} OR {})",
                folded_like("p.provider_type", term),
                folded_like("s.name", term)
            ));
        }
    }

    let query = format!(
        r#"SELECT p.company_name, p.provider_type, p.city, p.phone, s.name as service, s.movilo_price, s.original_price
           FROM movilo_providers p 
           LEFT JOIN movilo_provider_services s ON p.id = s.provider_id 
           WHERE {} 
           ORDER BY p.company_name LIMIT 10"#,
        conditions.join(" AND ")
    );

    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "movilo",
            "query": query
        }),
    };

    let json_result = super::data::execute_memento_query_json(&memento_call).await;

    match json_result {
        Err(error) => {
            // Keep the real error in the logs, never in the user-facing channel:
            // the tool output is the only path to the user and a weak model echoes
            // it verbatim, so it must be clean (the persona forbids leaking SQL /
            // internals). See format_provider_list for the streaming rationale.
            tracing::warn!("movilo_search_providers query failed: {error}");
            ToolResult {
                name: "movilo_search_providers".to_string(),
                success: false,
                output: "No pude consultar el directorio en este momento. Puedes buscarlo aquí: [Ver todo el directorio](https://movilo.club/#providers)".to_string(),
            }
        }
        Ok(res) => {
            let count = res.get("count").and_then(|c| c.as_i64()).unwrap_or(0);
            let rows = res
                .get("rows")
                .and_then(|r| r.as_array())
                .cloned()
                .unwrap_or_default();

            let mut widget_attrs = String::new();
            if !specialty.is_empty() {
                let canonical = canonical_widget_category(specialty);
                widget_attrs.push_str(&format!(" category=\"{}\"", escape_attr(canonical)));
            }
            if !keyword.is_empty() {
                widget_attrs.push_str(&format!(" search=\"{}\"", escape_attr(keyword)));
            } else if !specialty.is_empty()
                && !matches!(
                    canonical_widget_category(specialty),
                    "Clínica" | "Odontólogo" | "Laboratorio" | "Farmacia" | "Veterinaria"
                )
            {
                widget_attrs.push_str(&format!(" search=\"{}\"", escape_attr(specialty)));
            }
            // The <os-provider-map> tag is the interactive map the persona is told
            // to keep verbatim at the end of the reply. The output below is already
            // user-presentable end-to-end: no JSON, no SQL, no directive framing.
            let widget = format!("<os-provider-map{widget_attrs}></os-provider-map>");

            if count == 0 {
                let what = if !specialty.is_empty() {
                    specialty
                } else if !keyword.is_empty() {
                    keyword
                } else {
                    "prestadores"
                };
                let output = format!(
                    "No encontré {what} en Cali ahora mismo. ¿Probamos con otra zona o especialidad? También puedes ver el directorio completo aquí: [Ver todo el directorio](https://movilo.club/#providers)\n\n{widget}"
                );
                ToolResult {
                    name: "movilo_search_providers".to_string(),
                    success: true,
                    output,
                }
            } else {
                let list = format_provider_list(&rows);
                let output = format!(
                    "Estos son los prestadores de la red Movilo que coinciden con tu búsqueda:\n\n{list}\n\n{widget}"
                );
                ToolResult {
                    name: "movilo_search_providers".to_string(),
                    success: true,
                    output,
                }
            }
        }
    }
}

pub(crate) async fn execute_movilo_get_plans(_call: &ToolCall) -> ToolResult {
    let query = "SELECT name, price_annual, price_monthly, discount_percentage, features \
                 FROM movilo_plans WHERE is_active = true ORDER BY sort_order"
        .to_string();
    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({ "app": "movilo", "query": query }),
    };
    match super::data::execute_memento_query_json(&memento_call).await {
        Err(error) => ToolResult {
            name: "movilo_get_plans".to_string(),
            success: false,
            output: format!(
                "[[SYSTEM DIRECTIVE]]: No se pudieron consultar los planes. Di EXACTAMENTE: \"No pude consultar los planes en este momento. Puedes verlos y comprarlos aquí: [Comprar o Renovar Plan](https://movilo.club/buy)\". No inventes precios.\n\nError: {error}"
            ),
        },
        Ok(res) => {
            let rows = res.get("rows").cloned().unwrap_or(serde_json::json!([]));
            let formatted = serde_json::to_string_pretty(&rows).unwrap_or_default();
            let output = format!(
                "Planes de Movilo (precios en COP):\n{formatted}\n\n[[SYSTEM DIRECTIVE]]: Preséntale al usuario los planes con su PRECIO ANUAL real (formatea en pesos, ej. $499.000 COP) y su % de descuento. Aclara que es PAGO ÚNICO ANUAL (no mensual). Para el plan Empresarial di que es a medida. Usa SOLO las cifras del resultado — nunca inventes. Cierra con el enlace [Comprar o Renovar Plan](https://movilo.club/buy)."
            );
            ToolResult {
                name: "movilo_get_plans".to_string(),
                success: true,
                output,
            }
        }
    }
}

pub(crate) async fn execute_movilo_check_affiliation(call: &ToolCall) -> ToolResult {
    let email = call
        .arguments
        .get("email")
        .and_then(|e| e.as_str())
        .unwrap_or("");
    let doc = call
        .arguments
        .get("document")
        .and_then(|d| d.as_str())
        .unwrap_or("");

    if email.is_empty() && doc.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Debes proveer un email o documento para buscar la afiliación.".into(),
        };
    }

    let mut conditions = vec![];
    if !email.is_empty() {
        conditions.push(format!("email = '{}'", email.replace("'", "''")));
    }
    if !doc.is_empty() {
        // Fallback: Si existe campo de documento en la tabla (asumiremos que existe o buscaremos name)
        conditions.push(format!("id = '{}'", doc.replace("'", "''")));
    }

    let query = format!(
        "SELECT id, name, email, status, plan FROM movilo_users WHERE {} LIMIT 1",
        conditions.join(" OR ")
    );

    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "movilo",
            "query": query
        }),
    };
    execute_memento_query(&memento_call).await
}

pub(crate) async fn execute_movilo_validate_qr(call: &ToolCall) -> ToolResult {
    let qr_content = call
        .arguments
        .get("qr_content")
        .and_then(|q| q.as_str())
        .unwrap_or("");

    if qr_content.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "QRCode content is missing.".into(),
        };
    }

    // Asumimos que el QR emitido por Movilo tiene el User UUID o el Email
    let query = format!(
        "SELECT id, name, email, status, plan FROM movilo_users WHERE id = '{}' OR email = '{}' LIMIT 1",
        qr_content.replace("'", "''"),
        qr_content.replace("'", "''")
    );

    let memento_call = ToolCall {
        name: "memento_query".to_string(),
        arguments: serde_json::json!({
            "app": "movilo",
            "query": query
        }),
    };

    let db_result = execute_memento_query(&memento_call).await;
    if db_result.success && db_result.output.contains("rows") && !db_result.output.contains("[]") {
        ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "¡QR Validado Exitosamente! Datos del afiliado recuperados:\n{}",
                db_result.output
            ),
        }
    } else {
        ToolResult {
            name: call.name.clone(),
            success: false,
            output: "QR Inválido o usuario no encontrado en la base de datos de Movilo.".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{canonical_widget_category, escape_attr};

    #[test]
    fn maps_specific_medical_specialties_to_specialist_tab() {
        assert_eq!(canonical_widget_category("Dermatólogo"), "Especialista");
        assert_eq!(canonical_widget_category("Cardiología"), "Especialista");
    }

    #[test]
    fn preserves_explicit_non_specialist_categories() {
        assert_eq!(canonical_widget_category("Odontólogos"), "Odontólogo");
        assert_eq!(canonical_widget_category("Farmacia"), "Farmacia");
        assert_eq!(canonical_widget_category("Veterinaria"), "Veterinaria");
    }

    #[test]
    fn escapes_widget_attributes_as_html() {
        assert_eq!(
            escape_attr("Dermatólogo \"Norte\" & Cali"),
            "Dermatólogo &quot;Norte&quot; &amp; Cali"
        );
    }
}
