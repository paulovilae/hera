use std::env;
use std::io::{self, BufRead, IsTerminal, Read, Write};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

const DEFAULT_SOCKET: &str = "/tmp/hera-core.sock";

#[derive(Debug, Clone)]
struct Config {
    app: String,
    system: Option<String>,
    prompt: Option<String>,
    stream: bool,
    json_mode: bool,
    model: Option<String>,
    socket_path: String,
    permissions: Vec<String>,
    session: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            app: "claude-cli".to_string(),
            system: None,
            prompt: None,
            stream: true,
            json_mode: false,
            model: None,
            socket_path: DEFAULT_SOCKET.to_string(),
            permissions: Vec::new(),
            session: None,
        }
    }
}

#[derive(Debug)]
struct IpcReply {
    text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args(env::args().skip(1).collect())?;

    if let Some(prompt) = config.prompt.clone() {
        run_one_shot(&config, &prompt).await?;
        return Ok(());
    }

    if !io::stdin().is_terminal() {
        let mut stdin_buf = String::new();
        io::stdin().read_to_string(&mut stdin_buf)?;
        let prompt = stdin_buf.trim().to_string();
        if !prompt.is_empty() {
            run_one_shot(&config, &prompt).await?;
        }
        return Ok(());
    }

    run_repl(&config).await?;
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut free_args = Vec::new();
    let mut iter = args.into_iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-p" | "--print" | "--prompt" => {
                config.prompt = Some(iter.next().ok_or("missing value after --prompt")?);
            }
            "--system" => {
                config.system = Some(iter.next().ok_or("missing value after --system")?);
            }
            "--app" => {
                config.app = iter.next().ok_or("missing value after --app")?;
            }
            "--model" => {
                config.model = Some(iter.next().ok_or("missing value after --model")?);
            }
            "--socket" => {
                config.socket_path = iter.next().ok_or("missing value after --socket")?;
            }
            "--permission" => {
                config
                    .permissions
                    .push(iter.next().ok_or("missing value after --permission")?);
            }
            "--session" => {
                config.session = Some(iter.next().ok_or("missing value after --session")?);
            }
            "--json" => {
                config.json_mode = true;
                config.stream = false;
            }
            "--no-stream" => {
                config.stream = false;
            }
            "--stream" => {
                config.stream = true;
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown flag: {arg}").into());
            }
            _ => free_args.push(arg),
        }
    }

    if config.prompt.is_none() && !free_args.is_empty() {
        config.prompt = Some(free_args.join(" "));
    }

    Ok(config)
}

async fn run_one_shot(config: &Config, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let response = if config.stream {
        send_prompt(config, prompt).await?
    } else {
        send_prompt(config, prompt).await?
    };

    if !config.stream {
        println!("{}", response.text);
    }

    Ok(())
}

async fn run_repl(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut history: Vec<Value> = Vec::new();

    println!("Hera Claude CLI");
    println!("Type `/exit` to quit, `/clear` to reset history.");

    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            println!();
            break;
        }

        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if matches!(prompt, "/exit" | "/quit") {
            break;
        }
        if prompt == "/clear" {
            history.clear();
            println!("history cleared");
            continue;
        }

        history.push(json!({
            "role": "user",
            "content": prompt,
        }));

        let response = send_messages(config, &history).await?;
        println!();
        history.push(json!({
            "role": "assistant",
            "content": response.text,
        }));
    }

    Ok(())
}

async fn send_prompt(
    config: &Config,
    prompt: &str,
) -> Result<IpcReply, Box<dyn std::error::Error>> {
    let mut payload = json!({
        "app": config.app,
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 1200,
        "json_mode": config.json_mode,
        "permissions": config.permissions,
    });

    if let Some(system) = &config.system {
        payload["messages"] = json!([
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]);
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("prompt");
        }
    }

    if let Some(model) = &config.model {
        payload["model"] = json!(model);
    }
    apply_session(&mut payload, config);

    send_request(config, payload).await
}

/// Attach a per-call session identity so memory does not bleed across runs
/// (e.g. eval tasks): a unique session_id makes Hera derive a unique user_id,
/// so Memento recall stays scoped to this single conversation.
fn apply_session(payload: &mut Value, config: &Config) {
    if let Some(session) = &config.session
        && let Some(obj) = payload.as_object_mut()
    {
        obj.insert("session_id".to_string(), json!(session));
        obj.insert("chat_id".to_string(), json!(session));
        obj.insert("sender_name".to_string(), json!(session));
    }
}

async fn send_messages(
    config: &Config,
    history: &[Value],
) -> Result<IpcReply, Box<dyn std::error::Error>> {
    let mut messages = Vec::new();
    if let Some(system) = &config.system {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }
    messages.extend(history.iter().cloned());

    let mut payload = json!({
        "app": config.app,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1200,
        "json_mode": config.json_mode,
        "permissions": config.permissions,
    });

    if let Some(model) = &config.model {
        payload["model"] = json!(model);
    }
    apply_session(&mut payload, config);

    send_request(config, payload).await
}

async fn send_request(
    config: &Config,
    payload: Value,
) -> Result<IpcReply, Box<dyn std::error::Error>> {
    let action = if config.stream {
        "generate_stream"
    } else {
        "generate"
    };
    let request = json!({
        "action": action,
        "payload": payload,
    });

    let mut stream = UnixStream::connect(&config.socket_path).await?;
    let wire = serde_json::to_vec(&request)?;
    stream.write_all(&wire).await?;
    stream.shutdown().await?;

    let mut buffer = String::new();
    stream.read_to_string(&mut buffer).await?;
    parse_response(&buffer, config.stream)
}

fn parse_response(response: &str, streamed: bool) -> Result<IpcReply, Box<dyn std::error::Error>> {
    let mut text = String::new();
    let mut non_stream_result: Option<String> = None;

    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let msg: Value = serde_json::from_str(line)?;
        let status = msg.get("status").and_then(Value::as_str).unwrap_or("");

        match status {
            "stream_start" | "tool_status" | "done" => {}
            "chunk" => {
                if let Some(chunk) = msg.pointer("/data/text").and_then(Value::as_str) {
                    let cleaned = strip_think_prefix(chunk);
                    print!("{cleaned}");
                    io::stdout().flush()?;
                    text.push_str(cleaned);
                }
            }
            "success" => {
                if let Some(result) = msg.pointer("/data/result").and_then(Value::as_str) {
                    non_stream_result = Some(strip_think_prefix(result).to_string());
                }
            }
            "error" => {
                let error = msg
                    .pointer("/data/error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown Hera IPC error");
                return Err(error.to_string().into());
            }
            _ => {}
        }
    }

    if streamed {
        if !text.is_empty() {
            println!();
        }
        return Ok(IpcReply { text });
    }

    Ok(IpcReply {
        text: non_stream_result.unwrap_or_default(),
    })
}

fn strip_think_prefix(raw: &str) -> &str {
    if let Some(idx) = raw.rfind("</think>") {
        return raw[idx + "</think>".len()..].trim_start();
    }
    if let Some(idx) = raw.rfind("</thinking>") {
        return raw[idx + "</thinking>".len()..].trim_start();
    }
    raw
}

fn print_help() {
    println!(
        "Usage: claude [options] [prompt]\n\
         \n\
         Options:\n\
           -p, --prompt <text>       Run a one-shot prompt\n\
           --system <text>           Inject a system prompt\n\
           --app <name>              Hera app name for routing and policy\n\
           --model <name>            Override model hint\n\
           --socket <path>           Hera IPC socket path\n\
           --permission <name>       Allow a Hera tool (repeatable)\n\
           --stream                  Stream output (default)\n\
           --no-stream               Wait for the full response\n\
           --json                    Request raw JSON output\n\
           -h, --help                Show this help\n\
         \n\
         Examples:\n\
           claude -p \"Summarize this repo\"\n\
           claude --app vetra --system \"You are a contract assistant\"\n\
           echo \"List the open risks\" | claude --no-stream"
    );
}
