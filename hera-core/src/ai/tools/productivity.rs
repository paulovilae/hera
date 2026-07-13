//! Productivity tools: email reading, calendar, notes, persistent memory.

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{error, info, warn};

use crate::ai::tool_executor::{ToolCall, ToolResult};

const MEMENTO_SOCK: &str = "/tmp/memento.sock";
pub(crate) const IMAP_CONF: &str = "/home/paulo/.config/imagineos/secrets/imap.conf";

const CALENDAR_CONF: &str = "/home/paulo/.config/imagineos/secrets/calendar.conf";
const NOTES_DIR: &str = "/home/paulo/Notes";

// ─── Memento IPC helper ────────────────────────────────────────────────────

async fn memento_send(action: &str, payload: Value) -> Result<Value, String> {
    let mut stream = tokio::net::UnixStream::connect(MEMENTO_SOCK)
        .await
        .map_err(|e| format!("Memento not running: {}", e))?;

    let msg = serde_json::json!({ "action": action, "payload": payload });
    stream
        .write_all(msg.to_string().as_bytes())
        .await
        .map_err(|e| format!("Memento write error: {}", e))?;
    let _ = stream.shutdown().await;

    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("Memento read error: {}", e))?;

    serde_json::from_slice(&buf).map_err(|e| format!("Memento parse error: {}", e))
}

// ─── document_to_text ──────────────────────────────────────────────────────
// Conversor de documentos: Hera EXPONE el tool, pero el trabajo lo hace Memento (dueño de la
// ingesta de conocimiento). REGLA DE PLATAFORMA: se manda la UBICACIÓN, no el archivo. Reenvía a la
// acción IPC `extract_text {path}` de Memento y devuelve el texto extraído.

pub(crate) async fn execute_document_to_text(call: &ToolCall) -> ToolResult {
    let path = match call.arguments.get("path").and_then(|v| v.as_str()) {
        Some(p) if !p.trim().is_empty() => p.trim().to_string(),
        _ => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: "Missing 'path' argument".to_string(),
            }
        }
    };

    match memento_send("extract_text", serde_json::json!({ "path": path })).await {
        Ok(resp) => {
            if let Some(e) = resp.get("error").and_then(|v| v.as_str()) {
                return ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: format!("extract_text: {e}"),
                };
            }
            let text = resp.get("text").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            let source_type = resp.get("source_type").and_then(|v| v.as_str()).unwrap_or("text");
            if text.is_empty() {
                ToolResult {
                    name: call.name.clone(),
                    success: false,
                    output: "El documento no tiene texto extraíble (¿escaneado o vacío?).".to_string(),
                }
            } else {
                ToolResult {
                    name: call.name.clone(),
                    success: true,
                    output: format!("[{source_type}]\n{text}"),
                }
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Memento no disponible: {e}"),
        },
    }
}

// ─── save_memory ─────────────────────────────────────────────────────────

pub(crate) async fn execute_save_memory(call: &ToolCall) -> ToolResult {
    let content = match call.arguments.get("content").and_then(|v| v.as_str()) {
        Some(c) if !c.is_empty() => c.to_string(),
        _ => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: "Missing 'content' argument".to_string(),
            }
        }
    };

    let memory_type = call
        .arguments
        .get("memory_type")
        .and_then(|v| v.as_str())
        .unwrap_or("note")
        .to_string();

    let entry_title = call
        .arguments
        .get("entry_title")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let tags: Vec<Value> = call
        .arguments
        .get("tags")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_else(|| vec![Value::String(memory_type.clone())]);

    let expires_at = call.arguments.get("expires_at").cloned();

    let mut payload = serde_json::json!({
        "user_id": "paulo",
        "app_id": "ava",
        "expert_id": "ava",
        "content": content,
        "memory_type": memory_type,
        "entry_title": entry_title,
        "tags": tags,
        "status": "active",
        "auto_derive": false
    });

    if let Some(exp) = expires_at {
        payload["expires_at"] = exp;
    }

    info!(
        "💾 [Productivity] Saving memory type='{}' title='{}'",
        memory_type, entry_title
    );

    match memento_send("save_scoped_memory", payload).await {
        Ok(res) => {
            let id = res
                .get("id")
                .or_else(|| res.get("record_id"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            ToolResult {
                name: call.name.clone(),
                success: true,
                output: format!(
                    "Saved {} '{}' (id: {})",
                    memory_type,
                    if entry_title.is_empty() {
                        &content[..content.len().min(60)]
                    } else {
                        &entry_title
                    },
                    id
                ),
            }
        }
        Err(e) => {
            error!("save_memory failed: {}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Failed to save memory: {}", e),
            }
        }
    }
}

// ─── query_memory ─────────────────────────────────────────────────────────

pub(crate) async fn execute_query_memory(call: &ToolCall) -> ToolResult {
    let memory_type = call
        .arguments
        .get("memory_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let search = call
        .arguments
        .get("search")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let limit = call
        .arguments
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(20);

    let include_completed = call
        .arguments
        .get("include_completed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if let Some(kw) = &search {
        // Use search_memory_records for keyword queries
        let mut payload = serde_json::json!({
            "user_id": "paulo",
            "query": kw,
            "limit": limit
        });
        if let Some(mt) = &memory_type {
            payload["memory_type"] = Value::String(mt.clone());
        }
        if !include_completed {
            payload["status"] = Value::String("active".to_string());
        }

        return match memento_send("search_memory_records", payload).await {
            Ok(res) => format_memory_results(call, res),
            Err(e) => ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Query failed: {}", e),
            },
        };
    }

    let mut payload = serde_json::json!({
        "user_id": "paulo",
        "limit": limit
    });
    if let Some(mt) = &memory_type {
        payload["memory_type"] = Value::String(mt.clone());
    }
    if !include_completed {
        payload["status"] = Value::String("active".to_string());
    }

    match memento_send("query_memory_records", payload).await {
        Ok(res) => format_memory_results(call, res),
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Query failed: {}", e),
        },
    }
}

fn format_memory_results(call: &ToolCall, res: Value) -> ToolResult {
    let entries = res
        .get("entries")
        .or_else(|| res.get("results"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if entries.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: "No memory records found.".to_string(),
        };
    }

    let lines: Vec<String> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let title = e
                .get("entry_title")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let content = e
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mtype = e
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("note");
            let status = e
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("active");
            let ts = e
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let display = if !title.is_empty() {
                format!("{} — {}", title, &content[..content.len().min(80)])
            } else {
                content[..content.len().min(100)].to_string()
            };
            format!(
                "{}. [{}] [{}] {} ({})",
                i + 1,
                mtype,
                status,
                display,
                &ts[..ts.len().min(10)]
            )
        })
        .collect();

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("Found {} record(s):\n{}", entries.len(), lines.join("\n")),
    }
}

// ─── recall_session_context ───────────────────────────────────────────────

pub(crate) async fn execute_recall_session_context(call: &ToolCall) -> ToolResult {
    let app_id = call
        .arguments
        .get("app_id")
        .and_then(|v| v.as_str())
        .unwrap_or("ava")
        .to_string();

    let session_id = call
        .arguments
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let focus = call
        .arguments
        .get("focus")
        .and_then(|v| v.as_str())
        .unwrap_or("full")
        .to_string();

    let mut payload = serde_json::json!({
        "user_id": "paulo",
        "app_id": app_id
    });
    if let Some(sid) = session_id {
        payload["session_id"] = Value::String(sid);
    }

    info!(
        "[Productivity] recall_session_context app='{}' focus='{}'",
        app_id, focus
    );

    match memento_send("recall_recursive_context", payload).await {
        Ok(res) => {
            let mut parts: Vec<String> = Vec::new();

            // Durable facts — highest signal, always shown
            if let Some(facts) = res.get("durable_facts").and_then(|v| v.as_array()) {
                if !facts.is_empty() {
                    let lines: Vec<String> = facts
                        .iter()
                        .filter_map(|f| {
                            let content = f.get("content").and_then(|v| v.as_str())?;
                            let title = f
                                .get("entry_title")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            Some(if title.is_empty() {
                                format!("• {}", content)
                            } else {
                                format!("• [{}] {}", title, content)
                            })
                        })
                        .collect();
                    parts.push(format!("## Durable facts\n{}", lines.join("\n")));
                }
            }

            // Recent events — shown for "full" and "recent"
            if focus != "decisions" {
                if let Some(events) = res.get("recent_events").and_then(|v| v.as_array()) {
                    let shown: Vec<String> = events
                        .iter()
                        .take(8)
                        .filter_map(|e| {
                            let content = e.get("content").and_then(|v| v.as_str())?;
                            let ts = e
                                .get("timestamp")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            Some(format!("• {} ({})", &content[..content.len().min(120)], &ts[..ts.len().min(10)]))
                        })
                        .collect();
                    if !shown.is_empty() {
                        parts.push(format!("## Recent events\n{}", shown.join("\n")));
                    }
                }
            }

            // Session/project summaries — only for "full"
            if focus == "full" {
                for key in &["project_summaries", "room_summaries", "session_summaries"] {
                    if let Some(summaries) = res.get(key).and_then(|v| v.as_array()) {
                        if let Some(latest) = summaries.last() {
                            if let Some(content) = latest.get("content").and_then(|v| v.as_str()) {
                                let label = key.replace('_', " ");
                                parts.push(format!(
                                    "## {}\n{}",
                                    label,
                                    &content[..content.len().min(400)]
                                ));
                            }
                        }
                    }
                }
            }

            if parts.is_empty() {
                ToolResult {
                    name: call.name.clone(),
                    success: true,
                    output: format!("No prior context found for app='{}'.", app_id),
                }
            } else {
                ToolResult {
                    name: call.name.clone(),
                    success: true,
                    output: parts.join("\n\n"),
                }
            }
        }
        Err(e) => {
            warn!("recall_session_context failed: {}", e);
            ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Memento recall failed: {}", e),
            }
        }
    }
}

// ─── read_email ───────────────────────────────────────────────────────────

struct ImapCreds {
    host: String,
    port: u16,
    username: String,
    password: String,
    label: String,
}

const IMAP_SECRETS_DIR: &str = "/home/paulo/.config/imagineos/secrets";

fn parse_imap_conf(path: &str, label: &str) -> Option<ImapCreds> {
    let conf = std::fs::read_to_string(path).ok()?;
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
    let is_placeholder = password.starts_with("REPLACE_WITH") || password == "your_app_password";
    if host.is_empty() || username.is_empty() || password.is_empty() || is_placeholder {
        return None;
    }
    Some(ImapCreds { host, port, username, password, label: label.to_string() })
}

/// Discover all valid IMAP configs in the secrets dir.
/// imap.conf is primary (sovereign); imap-*.conf are additional inboxes.
fn discover_imap_configs() -> Vec<ImapCreds> {
    let mut configs = Vec::new();
    // Primary first
    if let Some(c) = parse_imap_conf(IMAP_CONF, "Soberano") {
        configs.push(c);
    }
    // Additional: imap-*.conf (alphabetical)
    if let Ok(entries) = std::fs::read_dir(IMAP_SECRETS_DIR) {
        let mut extras: Vec<_> = entries
            .flatten()
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                s.starts_with("imap-") && s.ends_with(".conf")
            })
            .collect();
        extras.sort_by_key(|e| e.file_name());
        for entry in extras {
            let path = entry.path();
            let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let label = stem.trim_start_matches("imap-").to_string();
            if let Some(c) = parse_imap_conf(path.to_str().unwrap_or(""), &label) {
                configs.push(c);
            }
        }
    }
    configs
}

static EMAIL_SCRIPT: &str = r#"
import imaplib, email, ssl, os
from email.header import decode_header

host   = os.environ['IMAP_HOST']
port   = int(os.environ.get('IMAP_PORT', '993'))
user   = os.environ['IMAP_USER']
passwd = os.environ['IMAP_PASS']
folder = os.environ.get('IMAP_FOLDER', 'INBOX')
limit  = int(os.environ.get('IMAP_LIMIT', '10'))
criteria = os.environ.get('IMAP_SEARCH', 'ALL')
source = os.environ.get('IMAP_SOURCE', '')

def decode_str(s):
    if s is None: return ''
    parts = decode_header(s)
    result = []
    for part, enc in parts:
        if isinstance(part, bytes):
            result.append(part.decode(enc or 'utf-8', errors='replace'))
        else:
            result.append(str(part))
    return ''.join(result)

def get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get('Content-Disposition', ''))
            if ct == 'text/plain' and 'attachment' not in cd:
                try:
                    payload = part.get_payload(decode=True)
                    return payload.decode(part.get_content_charset() or 'utf-8', errors='replace')[:500]
                except: pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                return payload.decode(msg.get_content_charset() or 'utf-8', errors='replace')[:500]
        except: pass
    return '(no text body)'

context = ssl.create_default_context()
try:
    mail = imaplib.IMAP4_SSL(host, port, ssl_context=context)
    mail.login(user, passwd)
    typ, sel_data = mail.select(folder)
    if criteria == 'ALL':
        # SEARCH ALL on a large mailbox (Gmail: tens of thousands of msgs)
        # returns a response over imaplib's line-length limit and raises.
        # Skip SEARCH entirely and fetch the last `limit` sequence numbers.
        count = int(sel_data[0])
        start = max(1, count - limit + 1)
        ids = [str(i).encode() for i in range(start, count + 1)]
    elif criteria.upper().startswith('X-GM-RAW'):
        # Gmail power-search: X-GM-RAW takes the raw Gmail query (e.g.
        # 'from:x has:attachment newer_than:7d') as a SEPARATE, quoted arg.
        # Passing it as one combined string makes imaplib emit a malformed
        # SEARCH -> Gmail replies 'Unknown argument X-GM-RAW'. Only Gmail
        # hosts support this extension; on other IMAP servers we return no
        # match rather than a garbage literal TEXT search.
        raw = criteria[len('X-GM-RAW'):].strip().strip('"')
        if 'gmail' in host.lower():
            _, data = mail.search(None, 'X-GM-RAW', '"%s"' % raw)
            ids = data[0].split()
            ids = ids[-limit:]
        else:
            ids = []
    else:
        _, data = mail.search(None, criteria)
        ids = data[0].split()
        ids = ids[-limit:]
    ids.reverse()
    results = []
    for uid in ids:
        _, msg_data = mail.fetch(uid, '(RFC822)')
        for part in msg_data:
            if isinstance(part, tuple):
                msg = email.message_from_bytes(part[1])
                msg_id = msg.get('Message-ID', '')
                subject = decode_str(msg.get('Subject', '(no subject)'))
                from_ = decode_str(msg.get('From', ''))
                date_ = msg.get('Date', '')
                body = get_body(msg)
                src_line = f"Inbox: {source}\n" if source else ''
                results.append(f"{src_line}UID: {uid.decode()}\nMessageID: {msg_id}\nFrom: {from_}\nSubject: {subject}\nDate: {date_}\n{body[:300]}\n---")
    mail.logout()
    print('\n'.join(results) if results else f'No messages found in {source or folder}.')
except Exception as e:
    print(f'ERROR [{source}]: {e}')
"#;

async fn fetch_inbox(tool_name: &str, creds: &ImapCreds, folder: &str, limit: i64, search: &str) -> String {
    let result = run_python_with_env(
        tool_name,
        EMAIL_SCRIPT,
        &[
            ("IMAP_HOST", creds.host.as_str()),
            ("IMAP_PORT", &creds.port.to_string()),
            ("IMAP_USER", creds.username.as_str()),
            ("IMAP_PASS", creds.password.as_str()),
            ("IMAP_FOLDER", folder),
            ("IMAP_LIMIT", &limit.to_string()),
            ("IMAP_SEARCH", search),
            ("IMAP_SOURCE", creds.label.as_str()),
        ],
    )
    .await;
    result.output
}

pub(crate) async fn execute_read_email(call: &ToolCall) -> ToolResult {
    let folder = call
        .arguments
        .get("folder")
        .and_then(|v| v.as_str())
        .unwrap_or("INBOX")
        .to_string();
    let limit = call
        .arguments
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(10);
    let unread_only = call
        .arguments
        .get("unread_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let search_term = call
        .arguments
        .get("search")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let search_criteria = if !search_term.is_empty() {
        search_term
    } else if unread_only {
        "UNSEEN".to_string()
    } else {
        "ALL".to_string()
    };

    let configs = discover_imap_configs();

    if configs.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!(
                "Incomplete credentials. Configure {}/imap.conf or {}/imap-<name>.conf files.",
                IMAP_SECRETS_DIR, IMAP_SECRETS_DIR
            ),
        };
    }

    info!("📧 [Productivity] Reading {} inbox(es)", configs.len());

    // Fetch all inboxes concurrently
    let fetches: Vec<_> = configs
        .iter()
        .map(|c| fetch_inbox(&call.name, c, &folder, limit, &search_criteria))
        .collect();
    let outputs = futures_util::future::join_all(fetches).await;

    let parts: Vec<String> = outputs.into_iter().filter(|s| !s.is_empty()).collect();

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: if parts.is_empty() {
            "No messages found in any inbox.".to_string()
        } else {
            parts.join("\n\n")
        },
    }
}

// ─── list_calendar_events ─────────────────────────────────────────────────

pub(crate) async fn execute_list_calendar_events(call: &ToolCall) -> ToolResult {
    let days_ahead = call
        .arguments
        .get("days_ahead")
        .and_then(|v| v.as_i64())
        .unwrap_or(7);
    let search = call
        .arguments
        .get("search")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let conf = match std::fs::read_to_string(CALENDAR_CONF) {
        Ok(c) => c,
        Err(_) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!(
                    "Calendar not configured. Create {} with:\nical_url=https://calendar.google.com/calendar/ical/YOUR_SECRET_URL/basic.ics\n\nGet this URL: Google Calendar → Settings → [calendar] → 'Secret address in iCal format'",
                    CALENDAR_CONF
                ),
            }
        }
    };

    let mut ical_url = String::new();
    for line in conf.lines() {
        if let Some(v) = line.trim().strip_prefix("ical_url=") {
            ical_url = v.trim().to_string();
        }
    }

    if ical_url.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("No ical_url found in {}.", CALENDAR_CONF),
        };
    }

    static CALENDAR_SCRIPT: &str = r#"
import urllib.request, ssl, re, os
from datetime import datetime, timezone, timedelta

url = os.environ['ICAL_URL']
days_ahead = int(os.environ.get('CAL_DAYS', '7'))
search_filter = os.environ.get('CAL_SEARCH', '')

context = ssl.create_default_context()
try:
    with urllib.request.urlopen(url, context=context, timeout=10) as resp:
        ical = resp.read().decode('utf-8', errors='replace')
except Exception as e:
    print(f'ERROR fetching calendar: {e}')
    exit(1)

def parse_dt(s):
    s = s.strip()
    if 'T' in s:
        s = re.sub(r';.*', '', s)
        try:
            if s.endswith('Z'):
                return datetime.strptime(s, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
            return datetime.strptime(s[:15], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
        except: return None
    else:
        s = re.sub(r';.*', '', s)[:8]
        try:
            return datetime.strptime(s, '%Y%m%d').replace(tzinfo=timezone.utc)
        except: return None

now = datetime.now(timezone.utc)
end = now + timedelta(days=days_ahead)

events = []
current = {}
for line in ical.splitlines():
    line = line.strip()
    if line == 'BEGIN:VEVENT':
        current = {}
    elif line == 'END:VEVENT':
        if current:
            events.append(current)
        current = {}
    elif line.startswith('SUMMARY'):
        current['summary'] = line.split(':', 1)[-1] if ':' in line else ''
    elif line.startswith('DTSTART'):
        current['dtstart'] = line.split(':', 1)[-1] if ':' in line else ''
    elif line.startswith('DTEND'):
        current['dtend'] = line.split(':', 1)[-1] if ':' in line else ''
    elif line.startswith('LOCATION'):
        current['location'] = line.split(':', 1)[-1] if ':' in line else ''
    elif line.startswith('DESCRIPTION'):
        current['description'] = (line.split(':', 1)[-1] if ':' in line else '')[:200]

results = []
for ev in events:
    dt = parse_dt(ev.get('dtstart', ''))
    if dt is None: continue
    if dt < now or dt > end: continue
    summary = ev.get('summary', '(no title)')
    if search_filter and search_filter.lower() not in summary.lower():
        continue
    loc = ev.get('location', '')
    desc = ev.get('description', '')
    dtend = parse_dt(ev.get('dtend', ''))
    time_str = dt.strftime('%a %b %d %H:%M UTC')
    if dtend:
        time_str += ' – ' + dtend.strftime('%H:%M')
    entry = f">> {summary}\n   {time_str}"
    if loc: entry += f"\n   @ {loc}"
    if desc: entry += f"\n   {desc[:100]}"
    results.append((dt, entry))

results.sort(key=lambda x: x[0])
if results:
    print('\n\n'.join(r[1] for r in results))
else:
    print(f'No events in the next {days_ahead} days.')
"#;

    info!("📅 [Productivity] Fetching calendar events ({} days)", days_ahead);
    run_python_with_env(
        &call.name,
        CALENDAR_SCRIPT,
        &[
            ("ICAL_URL", ical_url.as_str()),
            ("CAL_DAYS", &days_ahead.to_string()),
            ("CAL_SEARCH", search.as_str()),
        ],
    )
    .await
}

// ─── read_notes ───────────────────────────────────────────────────────────

pub(crate) async fn execute_read_notes(call: &ToolCall) -> ToolResult {
    let search = call
        .arguments
        .get("search")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let limit = call
        .arguments
        .get("limit")
        .and_then(|v| v.as_i64())
        .unwrap_or(10);
    let source = call
        .arguments
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("all")
        .to_string();

    let mut results: Vec<String> = Vec::new();

    // 1. Search Memento scoped memory for notes
    if source == "all" || source == "memory" {
        let mut payload = serde_json::json!({
            "user_id": "paulo",
            "memory_type": "note",
            "status": "active",
            "limit": limit
        });
        if !search.is_empty() {
            payload = serde_json::json!({
                "user_id": "paulo",
                "query": search,
                "memory_type": "note",
                "limit": limit
            });
        }

        let action = if search.is_empty() {
            "query_memory_records"
        } else {
            "search_memory_records"
        };

        match memento_send(action, payload).await {
            Ok(res) => {
                let entries = res
                    .get("entries")
                    .or_else(|| res.get("results"))
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                for e in &entries {
                    let title = e.get("entry_title").and_then(|v| v.as_str()).unwrap_or("");
                    let content = e.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    let ts = e.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
                    results.push(format!(
                        "[Memory] {} — {} ({})",
                        title,
                        &content[..content.len().min(200)],
                        &ts[..ts.len().min(10)]
                    ));
                }
            }
            Err(e) => warn!("read_notes memory query failed: {}", e),
        }
    }

    // 2. Search Memento knowledge store
    if source == "all" || source == "knowledge" {
        if !search.is_empty() {
            let payload = serde_json::json!({ "query": search });
            if let Ok(res) = memento_send("search_knowledge", payload).await {
                if let Some(items) = res.get("items").and_then(|v| v.as_array()) {
                    for item in items.iter().take(limit as usize) {
                        let key = item.get("key").and_then(|v| v.as_str()).unwrap_or("");
                        let content = item.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        results.push(format!(
                            "[Knowledge] {} — {}",
                            key,
                            &content[..content.len().min(200)]
                        ));
                    }
                }
            }
        } else {
            let payload = serde_json::json!({});
            if let Ok(res) = memento_send("list_knowledge", payload).await {
                if let Some(items) = res.get("items").and_then(|v| v.as_array()) {
                    for item in items.iter().take(limit as usize) {
                        let key = item.get("key").and_then(|v| v.as_str()).unwrap_or("");
                        let content = item.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        results.push(format!(
                            "[Knowledge] {} — {}",
                            key,
                            &content[..content.len().min(200)]
                        ));
                    }
                }
            }
        }
    }

    // 3. Scan local ~/Notes/ directory
    if source == "all" || source == "files" {
        if let Ok(entries) = std::fs::read_dir(NOTES_DIR) {
            let mut file_results: Vec<(String, String)> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .map(|x| x == "md" || x == "txt")
                        .unwrap_or(false)
                })
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let content = std::fs::read_to_string(e.path()).ok()?;
                    if !search.is_empty()
                        && !content.to_lowercase().contains(&search.to_lowercase())
                        && !name.to_lowercase().contains(&search.to_lowercase())
                    {
                        return None;
                    }
                    Some((name, content[..content.len().min(300)].to_string()))
                })
                .collect();
            file_results.truncate(limit as usize);
            for (name, content) in file_results {
                results.push(format!("[File] {} — {}", name, content.replace('\n', " ")));
            }
        }
    }

    if results.is_empty() {
        let msg = if search.is_empty() {
            "No notes found. You can create notes by asking Ava to save a note.".to_string()
        } else {
            format!("No notes found matching '{}'.", search)
        };
        return ToolResult {
            name: call.name.clone(),
            success: true,
            output: msg,
        };
    }

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: format!("Found {} note(s):\n\n{}", results.len(), results.join("\n\n")),
    }
}

// ─── Python execution helper ──────────────────────────────────────────────

/// Pick an IMAP account by label/username substring, or the primary (first) when
/// `account` is empty. Returns None if an explicit account was requested but not found.
fn pick_account<'a>(configs: &'a [ImapCreds], account: &str) -> Option<&'a ImapCreds> {
    if account.trim().is_empty() {
        return configs.first();
    }
    let a = account.to_lowercase();
    configs
        .iter()
        .find(|c| c.label.to_lowercase().contains(&a) || c.username.to_lowercase().contains(&a))
}

static DRAFT_SCRIPT: &str = r#"
import imaplib, ssl, os, time
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid

host   = os.environ['IMAP_HOST']
port   = int(os.environ.get('IMAP_PORT', '993'))
user   = os.environ['IMAP_USER']
passwd = os.environ['IMAP_PASS']
to      = os.environ.get('DRAFT_TO', '')
subject = os.environ.get('DRAFT_SUBJECT', '')
body    = os.environ.get('DRAFT_BODY', '')
folder  = os.environ.get('DRAFT_FOLDER', '').strip()

msg = MIMEText(body, 'plain', 'utf-8')
msg['From'] = user
if to:
    msg['To'] = to
msg['Subject'] = subject
msg['Date'] = formatdate(localtime=True)
msg['Message-ID'] = make_msgid()

context = ssl.create_default_context()
try:
    mail = imaplib.IMAP4_SSL(host, port, ssl_context=context)
    mail.login(user, passwd)
    if not folder:
        folder = '[Gmail]/Drafts' if 'gmail' in host.lower() else 'Drafts'
    r = mail.append(folder, '\\Draft', imaplib.Time2Internaldate(time.time()), msg.as_bytes())
    mail.logout()
    if r[0] == 'OK':
        print(f"Draft saved to '{folder}' on {user}\nTo: {to or '(none)'}\nSubject: {subject}")
    else:
        print(f'ERROR appending draft: {r}')
except Exception as e:
    print(f'ERROR: {e}')
"#;

static LABELS_SCRIPT: &str = r#"
import imaplib, ssl, os, re

host   = os.environ['IMAP_HOST']
port   = int(os.environ.get('IMAP_PORT', '993'))
user   = os.environ['IMAP_USER']
passwd = os.environ['IMAP_PASS']
source = os.environ.get('IMAP_SOURCE', '')

context = ssl.create_default_context()
try:
    mail = imaplib.IMAP4_SSL(host, port, ssl_context=context)
    mail.login(user, passwd)
    typ, data = mail.list()
    mail.logout()
    names = []
    for line in data or []:
        if isinstance(line, bytes):
            s = line.decode('utf-8', errors='replace')
            m = re.search(r'"([^"]+)"\s*$', s)
            names.append(m.group(1) if m else s.rsplit(' ', 1)[-1])
    hdr = f"Labels/folders on {source or user}:" if (source or user) else 'Labels/folders:'
    print(hdr + '\n' + ('\n'.join(names) if names else '(none)'))
except Exception as e:
    print(f'ERROR [{source}]: {e}')
"#;

/// `create_draft` — save an email draft to the account's Drafts folder via IMAP APPEND.
/// Sovereign (no send): the draft lands in Gmail/IMAP Drafts for the user to review + send.
pub(crate) async fn execute_create_draft(call: &ToolCall) -> ToolResult {
    let to = call.arguments.get("to").and_then(|v| v.as_str()).unwrap_or("");
    let subject = call.arguments.get("subject").and_then(|v| v.as_str()).unwrap_or("");
    let body = call.arguments.get("body").and_then(|v| v.as_str()).unwrap_or("");
    let account = call.arguments.get("account").and_then(|v| v.as_str()).unwrap_or("");
    let folder = call.arguments.get("folder").and_then(|v| v.as_str()).unwrap_or("");

    if subject.is_empty() || body.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing required fields: subject and body are required".to_string(),
        };
    }

    let configs = discover_imap_configs();
    let Some(creds) = pick_account(&configs, account) else {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: if account.is_empty() {
                format!("No IMAP account configured. Set up {}/imap.conf.", IMAP_SECRETS_DIR)
            } else {
                format!("No IMAP account matching '{}'. Available: {}", account,
                    configs.iter().map(|c| c.label.as_str()).collect::<Vec<_>>().join(", "))
            },
        };
    };

    info!("📧 [Productivity] Creating draft on {}", creds.label);
    run_python_with_env(
        &call.name,
        DRAFT_SCRIPT,
        &[
            ("IMAP_HOST", creds.host.as_str()),
            ("IMAP_PORT", &creds.port.to_string()),
            ("IMAP_USER", creds.username.as_str()),
            ("IMAP_PASS", creds.password.as_str()),
            ("DRAFT_TO", to),
            ("DRAFT_SUBJECT", subject),
            ("DRAFT_BODY", body),
            ("DRAFT_FOLDER", folder),
        ],
    )
    .await
}

async fn list_labels_for(tool_name: &str, creds: &ImapCreds) -> String {
    let port = creds.port.to_string();
    run_python_with_env(
        tool_name,
        LABELS_SCRIPT,
        &[
            ("IMAP_HOST", creds.host.as_str()),
            ("IMAP_PORT", port.as_str()),
            ("IMAP_USER", creds.username.as_str()),
            ("IMAP_PASS", creds.password.as_str()),
            ("IMAP_SOURCE", creds.label.as_str()),
        ],
    )
    .await
    .output
}

/// `list_labels` — list mailbox folders / Gmail labels via IMAP LIST, for every inbox.
pub(crate) async fn execute_list_labels(call: &ToolCall) -> ToolResult {
    let configs = discover_imap_configs();
    if configs.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("No IMAP account configured. Set up {}/imap.conf.", IMAP_SECRETS_DIR),
        };
    }

    let fetches: Vec<_> = configs
        .iter()
        .map(|c| list_labels_for(&call.name, c))
        .collect();
    let outputs = futures_util::future::join_all(fetches).await;
    let parts: Vec<String> = outputs.into_iter().filter(|s| !s.is_empty()).collect();

    ToolResult {
        name: call.name.clone(),
        success: true,
        output: parts.join("\n\n"),
    }
}

pub(crate) async fn run_python_with_env(tool_name: &str, script: &str, env: &[(&str, &str)]) -> ToolResult {
    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg("-c").arg(script);
    for (k, v) in env {
        cmd.env(k, v);
    }

    match cmd.output().await {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            if !out.status.success() || stdout.starts_with("ERROR") {
                let err = if !stderr.is_empty() { &stderr } else { &stdout };
                ToolResult {
                    name: tool_name.to_string(),
                    success: false,
                    output: format!("Tool error: {}", err.trim()),
                }
            } else {
                ToolResult {
                    name: tool_name.to_string(),
                    success: true,
                    output: stdout.trim().to_string(),
                }
            }
        }
        Err(e) => ToolResult {
            name: tool_name.to_string(),
            success: false,
            output: format!("Failed to launch Python: {}", e),
        },
    }
}
