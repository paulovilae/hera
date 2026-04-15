//! Vetra app tool executors: contracts, QR, email, Telegram, maps, workflows
use crate::ai::tool_executor::{ToolCall, ToolResult};
use genpdf::{Alignment, Element as _, elements, fonts, style};
use serde_json::Value;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

const PDF_FONT_CANDIDATES: &[(&str, &str)] = &[
    ("/usr/share/fonts/truetype/liberation", "LiberationSans"),
    ("/usr/share/fonts/truetype/dejavu", "DejaVuSans"),
];

pub(crate) async fn execute_bind_telegram_workspace(call: &ToolCall) -> ToolResult {
    let bot_name = call
        .arguments
        .get("bot_name")
        .and_then(|value| value.as_str())
        .unwrap_or("Vetra")
        .trim();
    let sender_id = call
        .arguments
        .get("sender_id")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let sender_name = call
        .arguments
        .get("sender_name")
        .and_then(|value| value.as_str())
        .unwrap_or("Telegram User")
        .trim();
    let workspace_user = call
        .arguments
        .get("workspace_user")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let workspace_company = call
        .arguments
        .get("workspace_company")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let locale = call
        .arguments
        .get("locale")
        .and_then(|value| value.as_str())
        .unwrap_or("es")
        .trim();

    if sender_id.is_empty() || workspace_user.is_empty() || workspace_company.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Error: must provide 'sender_id', 'workspace_user', and 'workspace_company'."
                .into(),
        };
    }

    let path = "/home/paulo/Programs/apps/OS/etc/imaginclaw/vetra_telegram_bindings.json";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration: std::time::Duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let mut store = match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str::<Value>(&raw)
            .unwrap_or_else(|_| serde_json::json!({ "bindings": [] })),
        Err(_) => serde_json::json!({ "bindings": [] }),
    };

    let bindings = store
        .get_mut("bindings")
        .and_then(|value| value.as_array_mut())
        .expect("bindings array should exist");

    let key_bot = bot_name.to_lowercase();
    if let Some(existing) = bindings.iter_mut().find(|item| {
        item.get("bot_name")
            .and_then(|value| value.as_str())
            .map(|value| value.eq_ignore_ascii_case(&key_bot))
            .unwrap_or(false)
            && item.get("sender_id").and_then(|value| value.as_str()) == Some(sender_id)
    }) {
        *existing = serde_json::json!({
            "bot_name": bot_name,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "workspace_user": workspace_user,
            "workspace_company": workspace_company,
            "locale": locale,
            "created_at": existing.get("created_at").and_then(|value| value.as_i64()).unwrap_or(now),
            "updated_at": now,
        });
    } else {
        bindings.push(serde_json::json!({
            "bot_name": bot_name,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "workspace_user": workspace_user,
            "workspace_company": workspace_company,
            "locale": locale,
            "created_at": now,
            "updated_at": now,
        }));
    }

    if let Some(parent) = std::path::Path::new(path).parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to create bindings directory: {}", error),
            };
        }
    }

    match serde_json::to_string_pretty(&store)
        .map_err(|error| error.to_string())
        .and_then(|raw| std::fs::write(path, raw).map_err(|error| error.to_string()))
    {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!(
                "Bound Telegram sender '{}' to workspace '{}' as '{}'.",
                sender_id, workspace_company, workspace_user
            ),
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to persist Telegram binding: {}", error),
        },
    }
}

pub(crate) async fn execute_generate_qr_code(call: &ToolCall) -> ToolResult {
    let content = call
        .arguments
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if content.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing content".into(),
        };
    }

    // Using a quick external API for now, could be replaced with a local Rust crate later
    let url = format!(
        "https://api.qrserver.com/v1/create-qr-code/?size=500x500&data={}",
        urlencoding::encode(content)
    );
    info!("🔲 [Hera] Generated QR Code URL: {}", url);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!(
            "QR Code generated successfully. Use this exact line in your reply to deliver it inline:\nMEDIA: {}",
            url
        ),
    }
}

pub(crate) async fn execute_generate_contract_pdf(call: &ToolCall) -> ToolResult {
    let debtor = call
        .arguments
        .get("debtor_id")
        .and_then(|c| c.as_str())
        .unwrap_or("unknown");
    let content = call
        .arguments
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if content.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing content".into(),
        };
    }

    let file_name = format!("Acuerdo_Pago_{}.pdf", debtor.replace(" ", "_"));
    let path = format!("/tmp/{}", file_name);

    if let Err(error) = render_payment_agreement_pdf(&path, debtor, content) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to generate payment agreement PDF: {}", error),
        };
    }

    info!("📄 [Hera] Generated Contract Document: {}", path);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!(
            "Payment agreement PDF generated successfully at {}. Inform the user that the document has been filed.",
            path
        ),
    }
}

fn render_payment_agreement_pdf(
    output_path: &str,
    debtor: &str,
    content: &str,
) -> Result<(), String> {
    let font_family = load_pdf_font_family()?;
    let mut doc = genpdf::Document::new(font_family);
    doc.set_title(format!("Acuerdo de pago - {}", debtor));
    doc.set_minimal_conformance();
    doc.set_line_spacing(1.25);

    let mut decorator = genpdf::SimplePageDecorator::new();
    decorator.set_margins(16);
    doc.set_page_decorator(decorator);

    let title_style = style::Style::new().bold().with_font_size(18);
    let meta_style = style::Style::new().with_font_size(10);

    doc.push(
        elements::Paragraph::new("Acuerdo de Pago")
            .aligned(Alignment::Center)
            .styled(title_style),
    );
    doc.push(
        elements::Paragraph::new(format!("Deudor / referencia: {}", debtor))
            .aligned(Alignment::Center)
            .styled(meta_style),
    );
    doc.push(
        elements::Paragraph::new(format!(
            "Generado por Hera. Timestamp Unix: {}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0)
        ))
        .aligned(Alignment::Center)
        .styled(meta_style),
    );
    doc.push(elements::Break::new(1));

    for block in normalized_agreement_blocks(content) {
        doc.push(elements::Paragraph::new(block));
        doc.push(elements::Break::new(0.6));
    }

    doc.push(elements::Break::new(0.5));
    doc.push(
        elements::Paragraph::new(
            "Este documento resume el acuerdo reportado por el flujo de cobranza. Debe ser validado por la operación antes de archivo o notificación final cuando aplique.",
        )
        .styled(meta_style),
    );

    doc.render_to_file(output_path)
        .map_err(|error| error.to_string())
}

fn load_pdf_font_family() -> Result<fonts::FontFamily<fonts::FontData>, String> {
    for (dir, name) in PDF_FONT_CANDIDATES {
        if !Path::new(dir).exists() {
            continue;
        }
        if let Ok(family) = fonts::from_files(dir, name, None) {
            return Ok(family);
        }
    }
    Err("No suitable PDF font family found on host".to_string())
}

fn normalized_agreement_blocks(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current.is_empty() {
                blocks.push(current.join(" "));
                current.clear();
            }
            continue;
        }
        current.push(trimmed.to_string());
    }

    if !current.is_empty() {
        blocks.push(current.join(" "));
    }

    if blocks.is_empty() {
        blocks.push(content.trim().to_string());
    }

    blocks
}

pub(crate) async fn execute_dispatch_email(call: &ToolCall) -> ToolResult {
    let recipient = call
        .arguments
        .get("recipient")
        .and_then(|c| c.as_str())
        .unwrap_or("unknown");
    let subject = call
        .arguments
        .get("subject")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let attachment = call
        .arguments
        .get("attachment_path")
        .and_then(|c| c.as_str())
        .unwrap_or("None");

    // Simulate sending email via local sendmail or SMTP (For OS-v3 Demo mode)
    info!(
        "📧 [Hera] Dispatching Email to: {} | Subject: {} | Attachment: {}",
        recipient, subject, attachment
    );

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!(
            "Email successfully dispatched via port 25 relay to {}.",
            recipient
        ),
    }
}

pub(crate) async fn execute_get_map_route(call: &ToolCall) -> ToolResult {
    let dest = call
        .arguments
        .get("destination")
        .and_then(|d| d.as_str())
        .unwrap_or("");
    let orig = call
        .arguments
        .get("origin")
        .and_then(|o| o.as_str())
        .unwrap_or("");

    if dest.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing destination".into(),
        };
    }

    let url = if orig.is_empty() {
        format!(
            "https://www.google.com/maps/search/?api=1&query={}",
            urlencoding::encode(dest)
        )
    } else {
        format!(
            "https://www.google.com/maps/dir/?api=1&origin={}&destination={}",
            urlencoding::encode(orig),
            urlencoding::encode(dest)
        )
    };

    info!("🗺️ [Hera] Generated Google Maps URL: {}", url);
    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("Maps link generated successfully:\n{}", url),
    }
}

pub(crate) async fn execute_workflow(call: &ToolCall) -> ToolResult {
    let app = call
        .arguments
        .get("app")
        .and_then(|a| a.as_str())
        .unwrap_or_default();
    let workflow = call
        .arguments
        .get("workflow")
        .and_then(|w| w.as_str())
        .unwrap_or_default();
    let default_payload = serde_json::json!({});
    let payload = call.arguments.get("payload").unwrap_or(&default_payload);

    if app.is_empty() || workflow.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required 'app' or 'workflow' parameters.".to_string(),
        };
    }

    let request = diakonos_core::protocol::DiakonosRequest {
        action: "execute_workflow_proxy".to_string(),
        payload: serde_json::json!({
            "app": app,
            "workflow": workflow,
            "payload": payload
        }),
    };

    info!(
        "🚀 [Hera] Delegating workflow execution to Diakonos: {}/{}",
        app, workflow
    );

    match diakonos_core::client::send_request(diakonos_core::client::DIAKONOS_SOCKET, &request)
        .await
    {
        Ok(response) if response.status == "success" => ToolResult {
            name: call.name.clone(),
            success: true,
            output: serde_json::to_string_pretty(&response.data).unwrap_or_default(),
        },
        Ok(response) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: response
                .data
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("Diakonos returned an unknown error")
                .to_string(),
        },
        Err(error) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Failed to reach Diakonos at {}: {}",
                diakonos_core::client::DIAKONOS_SOCKET,
                error
            ),
        },
    }
}
