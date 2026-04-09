//! Movilo app tool executors
use crate::ai::tool_executor::{ToolCall, ToolResult};
use crate::ai::tools::data::execute_memento_query;
use serde_json::Value;

pub(crate) async fn execute_movilo_search_providers(call: &ToolCall) -> ToolResult {
    let city = call
        .arguments
        .get("city")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let specialty = call
        .arguments
        .get("specialty")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let keyword = call
        .arguments
        .get("service_keyword")
        .and_then(|k| k.as_str())
        .unwrap_or("");

    let mut conditions = vec!["p.status = 'Aprobado'".to_string()];
    if !city.is_empty() {
        conditions.push(format!("p.city ILIKE '%{}%'", city.replace("'", "''")));
    }
    if !specialty.is_empty() {
        conditions.push(format!(
            "p.provider_type ILIKE '%{}%'",
            specialty.replace("'", "''")
        ));
    }
    if !keyword.is_empty() {
        conditions.push(format!("s.name ILIKE '%{}%'", keyword.replace("'", "''")));
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

    let mut result = execute_memento_query(&memento_call).await;

    // Instruct the AI to render the map component based on the search context
    if result.success {
        let mut widget_attrs = String::new();
        if !specialty.is_empty() {
            widget_attrs.push_str(&format!(
                " category=\"{}\"",
                specialty.replace("\"", "\\\"")
            ));
        }
        if !keyword.is_empty() {
            widget_attrs.push_str(&format!(" search=\"{}\"", keyword.replace("\"", "\\\"")));
        } else if !city.is_empty() {
            widget_attrs.push_str(&format!(" search=\"{}\"", city.replace("\"", "\\\"")));
        }

        result.output.push_str(&format!(
            "\n\n[[SYSTEM DIRECTIVE]]: You MUST also embed an interactive map in your response so the user can visually locate these providers. To do this, simply include the following EXACT string somewhere in your text reply:\n\nWIDGET: <os-provider-map{}></os-provider-map>\n",
            widget_attrs
        ));
    }

    result
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

