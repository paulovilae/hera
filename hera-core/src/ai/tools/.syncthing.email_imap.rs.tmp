//! Email IMAP tools: reply, mark_read, move, delete

use crate::ai::tool_executor::{ToolCall, ToolResult};
use super::productivity;
use serde_json::Value;
use tracing::info;

fn shared_secret() -> Option<String> {
    let path = std::env::var("OS_AUTH_SHARED_SECRET_FILE").unwrap_or_else(|_| {
        "/home/paulo/.config/imagineos/secrets/os-auth-shared-secret".to_string()
    });
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

pub(crate) async fn execute_reply_email(call: &ToolCall) -> ToolResult {
    let to = call
        .arguments
        .get("to")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let subject = call
        .arguments
        .get("subject")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let body = call
        .arguments
        .get("body")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let html = call
        .arguments
        .get("html")
        .and_then(|v| v.as_str());
    let in_reply_to = call
        .arguments
        .get("in_reply_to")
        .and_then(|v| v.as_str());
    let references = call
        .arguments
        .get("references")
        .and_then(|v| v.as_str());
    let app_slug = call
        .arguments
        .get("app_slug")
        .and_then(|v| v.as_str())
        .unwrap_or("hera");

    if to.is_empty() || subject.is_empty() || body.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required fields: to, subject, and body are required".to_string(),
        };
    }

    let Some(secret) = shared_secret() else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "No pude leer el secret compartido (OS_AUTH_SHARED_SECRET_FILE).".to_string(),
        };
    };

    let mut payload = serde_json::json!({
        "app_slug": app_slug,
        "to": to,
        "subject": subject,
        "body": body,
    });

    if let Some(h) = html {
        payload.as_object_mut().unwrap().insert("html".to_string(), Value::String(h.to_string()));
    }
    if let Some(irt) = in_reply_to {
        payload.as_object_mut().unwrap().insert("in_reply_to".to_string(), Value::String(irt.to_string()));
    }
    if let Some(refs) = references {
        payload.as_object_mut().unwrap().insert("references".to_string(), Value::String(refs.to_string()));
    }

    info!("📧 [EmailIMAP] Replying to email: {}", to);

    let client = reqwest::Client::new();
    let url = "http://127.0.0.1:5177/api/platform/email/send";

    match client
        .post(url)
        .header("x-os-service-token", secret)
        .json(&payload)
        .send()
        .await
    {
        Ok(resp) => {
            if resp.status().is_success() {
                match resp.json::<serde_json::Value>().await {
                    Ok(json) => {
                        if json.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                            let log_id = json
                                .get("data")
                                .and_then(|d| d.get("log_id"))
                                .and_then(|l| l.as_str())
                                .unwrap_or("unknown");
                            ToolResult {
                                name: call.name.clone(),
                                success: true,
                                output: format!("Reply sent successfully. Log ID: {}", log_id),
                            }
                        } else {
                            let err = json
                                .get("error")
                                .and_then(|e| e.as_str())
                                .unwrap_or("Unknown error");
                            ToolResult {
                                name: call.name.clone(),
                                success: false,
                                output: format!("Reply failed: {}", err),
                            }
                        }
                    }
                    Err(e) => ToolResult {
                        name: call.name.clone(),
                        success: false,
                        output: format!("Failed to parse response: {}", e),
                    },
                }
            } else {
                ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("HTTP error: {}", resp.status()),
                }
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to send HTTP request: {}", e),
        },
    }
}

pub(crate) async fn execute_mark_read(call: &ToolCall) -> ToolResult {
    let uid = call
        .arguments
        .get("uid")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let folder = call
        .arguments
        .get("folder")
        .and_then(|v| v.as_str())
        .unwrap_or("INBOX");

    if uid.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required field: uid".to_string(),
        };
    }

    // Read credentials
    let conf = match std::fs::read_to_string(productivity::IMAP_CONF) {
        Ok(c) => c,
        Err(_) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!(
                    "Email credentials not configured. Create {} with:\nhost=imap.gmail.com\nport=993\nusername=your@gmail.com\npassword=your_app_password",
                    productivity::IMAP_CONF
                ),
            }
        }
    };

    let mut host = String::new();
    let mut port = 993u16;
    let mut username = String::new();
    let mut password = String::new();
    for line in conf.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("host=") {
            host = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("port=") {
            port = v.trim().parse().unwrap_or(993);
        } else if let Some(v) = line.strip_prefix("username=") {
            username = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("password=") {
            password = v.trim().to_string();
        }
    }

    if host.is_empty() || username.is_empty() || password.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Incomplete credentials in {}. Need: host, port, username, password.",
                productivity::IMAP_CONF
            ),
        };
    }

    static MARK_READ_SCRIPT: &str = r#"
import imaplib, ssl, os

host   = os.environ['IMAP_HOST']
port   = int(os.environ.get('IMAP_PORT', '993'))
user   = os.environ['IMAP_USER']
passwd = os.environ['IMAP_PASS']
folder = os.environ.get('IMAP_FOLDER', 'INBOX')
uid    = os.environ['IMAP_UID']

context = ssl.create_default_context()
try:
    mail = imaplib.IMAP4_SSL(host, port, ssl_context=context)
    mail.login(user, passwd)
    mail.select(folder)
    mail.uid('store', uid, '+FLAGS', '\\Seen')
    mail.logout()
    print(f'Marked email {uid} as read in {folder}')
except Exception as e:
    print(f'ERROR: {e}')
"#;

    info!("📧 [EmailIMAP] Marking email {} as read in {}", uid, folder);

    productivity::run_python_with_env(
        &call.name,
        MARK_READ_SCRIPT,
        &[
            ("IMAP_HOST", &host),
            ("IMAP_PORT", &port.to_string()),
            ("IMAP_USER", &username),
            ("IMAP_PASS", &password),
            ("IMAP_FOLDER", folder),
            ("IMAP_UID", uid),
        ],
    )
    .await
}

pub(crate) async fn execute_move_email(call: &ToolCall) -> ToolResult {
    let uid = call
        .arguments
        .get("uid")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let from_folder = call
        .arguments
        .get("from_folder")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let to_folder = call
        .arguments
        .get("to_folder")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if uid.is_empty() || from_folder.is_empty() || to_folder.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required fields: uid, from_folder, and to_folder are required".to_string(),
        };
    }

    // Read credentials
    let conf = match std::fs::read_to_string(productivity::IMAP_CONF) {
        Ok(c) => c,
        Err(_) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!(
                    "Email credentials not configured. Create {} with:\nhost=imap.gmail.com\nport=993\nusername=your@gmail.com\npassword=your_app_password",
                    productivity::IMAP_CONF
                ),
            }
        }
    };

    let mut host = String::new();
    let mut port = 993u16;
    let mut username = String::new();
    let mut password = String::new();
    for line in conf.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("host=") {
            host = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("port=") {
            port = v.trim().parse().unwrap_or(993);
        } else if let Some(v) = line.strip_prefix("username=") {
            username = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("password=") {
            password = v.trim().to_string();
        }
    }

    if host.is_empty() || username.is_empty() || password.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Incomplete credentials in {}. Need: host, port, username, password.",
                productivity::IMAP_CONF
            ),
        };
    }

    static MOVE_EMAIL_SCRIPT: &str = r#"
import imaplib, ssl, os

host        = os.environ['IMAP_HOST']
port        = int(os.environ.get('IMAP_PORT', '993'))
user        = os.environ['IMAP_USER']
passwd      = os.environ['IMAP_PASS']
from_folder = os.environ['IMAP_FROM_FOLDER']
to_folder   = os.environ['IMAP_TO_FOLDER']
uid         = os.environ['IMAP_UID']

context = ssl.create_default_context()
try:
    mail = imaplib.IMAP4_SSL(host, port, ssl_context=context)
    mail.login(user, passwd)
    mail.select(from_folder)
    mail.uid('copy', uid, to_folder)
    mail.uid('store', uid, '+FLAGS', '\\Deleted')
    mail.expunge()
    mail.logout()
    print(f'Moved email {uid} from {from_folder} to {to_folder}')
except Exception as e:
    print(f'ERROR: {e}')
"#;

    info!("📧 [EmailIMAP] Moving email {} from {} to {}", uid, from_folder, to_folder);

    productivity::run_python_with_env(
        &call.name,
        MOVE_EMAIL_SCRIPT,
        &[
            ("IMAP_HOST", &host),
            ("IMAP_PORT", &port.to_string()),
            ("IMAP_USER", &username),
            ("IMAP_PASS", &password),
            ("IMAP_FROM_FOLDER", from_folder),
            ("IMAP_TO_FOLDER", to_folder),
            ("IMAP_UID", uid),
        ],
    )
    .await
}

pub(crate) async fn execute_delete_email(call: &ToolCall) -> ToolResult {
    let uid = call
        .arguments
        .get("uid")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let folder = call
        .arguments
        .get("folder")
        .and_then(|v| v.as_str())
        .unwrap_or("INBOX");

    if uid.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required field: uid".to_string(),
        };
    }

    // Read credentials
    let conf = match std::fs::read_to_string(productivity::IMAP_CONF) {
        Ok(c) => c,
        Err(_) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!(
                    "Email credentials not configured. Create {} with:\nhost=imap.gmail.com\nport=993\nusername=your@gmail.com\npassword=your_app_password",
                    productivity::IMAP_CONF
                ),
            }
        }
    };

    let mut host = String::new();
    let mut port = 993u16;
    let mut username = String::new();
    let mut password = String::new();
    for line in conf.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("host=") {
            host = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("port=") {
            port = v.trim().parse().unwrap_or(993);
        } else if let Some(v) = line.strip_prefix("username=") {
            username = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("password=") {
            password = v.trim().to_string();
        }
    }

    if host.is_empty() || username.is_empty() || password.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Incomplete credentials in {}. Need: host, port, username, password.",
                productivity::IMAP_CONF
            ),
        };
    }

    static DELETE_EMAIL_SCRIPT: &str = r#"
import imaplib, ssl, os

host   = os.environ['IMAP_HOST']
port   = int(os.environ.get('IMAP_PORT', '993'))
user   = os.environ['IMAP_USER']
passwd = os.environ['IMAP_PASS']
folder = os.environ.get('IMAP_FOLDER', 'INBOX')
uid    = os.environ['IMAP_UID']

context = ssl.create_default_context()
try:
    mail = imaplib.IMAP4_SSL(host, port, ssl_context=context)
    mail.login(user, passwd)
    mail.select(folder)
    mail.uid('store', uid, '+FLAGS', '\\Deleted')
    mail.expunge()
    mail.logout()
    print(f'Deleted email {uid} from {folder}')
except Exception as e:
    print(f'ERROR: {e}')
"#;

    info!("📧 [EmailIMAP] Deleting email {} from {}", uid, folder);

    productivity::run_python_with_env(
        &call.name,
        DELETE_EMAIL_SCRIPT,
        &[
            ("IMAP_HOST", &host),
            ("IMAP_PORT", &port.to_string()),
            ("IMAP_USER", &username),
            ("IMAP_PASS", &password),
            ("IMAP_FOLDER", folder),
            ("IMAP_UID", uid),
        ],
    )
    .await
}
