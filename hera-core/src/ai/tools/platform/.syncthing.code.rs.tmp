//! Code execution tool executors: run_code (Python + Rust beans) and write_file.
use crate::ai::tool_executor::{ToolCall, ToolResult};
use super::{resolve_guarded_fs_path, validate_python_package_name};

pub(crate) async fn execute_run_code(call: &ToolCall) -> ToolResult {
    let lang = call
        .arguments
        .get("language")
        .and_then(|l| l.as_str())
        .unwrap_or("python");
    let code = call
        .arguments
        .get("code")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let packages: Vec<String> = call
        .arguments
        .get("packages")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if code.trim().is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing code payload.".to_string(),
        };
    }
    if code.len() > 100_000 {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Code payload too large. Limit is 100000 bytes.".to_string(),
        };
    }
    if packages.len() > 16
        || packages
            .iter()
            .any(|pkg| !validate_python_package_name(pkg))
    {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Package list contains invalid names or exceeds the maximum allowed count."
                .to_string(),
        };
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let bean_name = format!("bean_{}", timestamp);
    let beans_dir = "/home/paulo/Programs/apps/OS/Beans";
    let _ = std::fs::create_dir_all(beans_dir);

    // Cognitive Memory Pipeline: Record bean logic into Memento universally before execution
    // Doing it natively blocking here; socket is fast UDS.
    if let Ok(mut stream) = std::os::unix::net::UnixStream::connect("/tmp/memento.sock") {
        use std::io::Write;
        let payload = serde_json::json!({
            "action": "store_knowledge",
            "payload": {
                "key": bean_name.clone(),
                "content": format!("Language: {}\nPackages: {:?}\nCode:\n{}", lang, packages, code),
                "tags": "bean, code_interpreter"
            },
            "client": {
                "app": "hera",
                "token": std::env::var("MEMENTO_CLIENT_TOKEN").ok()
            }
        });
        let _ = stream.write_all(payload.to_string().as_bytes());
    }

    if lang.to_lowercase() == "rust" {
        execute_rust_bean(call, &bean_name, beans_dir, code, &packages)
    } else if lang.to_lowercase() == "python" {
        execute_python_bean(call, &bean_name, beans_dir, code, &packages)
    } else {
        ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Language '{}' not supported. Use 'rust' or 'python'.", lang),
        }
    }
}

fn execute_rust_bean(
    call: &ToolCall,
    bean_name: &str,
    beans_dir: &str,
    code: &str,
    packages: &[String],
) -> ToolResult {
    let project_dir = format!("{}/{}", beans_dir, bean_name);
    if let Err(e) = std::fs::create_dir_all(format!("{}/src", project_dir)) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to create Rust bean directory: {}", e),
        };
    }

    let mut deps = r#"[dependencies]
tokio = { version = "1", features = ["full", "rt-multi-thread"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
"#
    .to_string();
    for pkg in packages {
        deps.push_str(&format!("{} = \"*\"\n", pkg));
    }

    let cargo_toml = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

{}
"#,
        bean_name, deps
    );

    if let Err(e) = std::fs::write(format!("{}/Cargo.toml", project_dir), cargo_toml) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write Cargo.toml: {}", e),
        };
    }
    if let Err(e) = std::fs::write(format!("{}/src/main.rs", project_dir), code) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write src/main.rs: {}", e),
        };
    }

    match std::process::Command::new("cargo")
        .arg("run")
        .arg("--release")
        .current_dir(&project_dir)
        .output()
    {
        Ok(out) => {
            let out_str = String::from_utf8_lossy(&out.stdout).to_string();
            let err_str = String::from_utf8_lossy(&out.stderr).to_string();
            let success = out.status.success();

            let mut final_out =
                format!("RUST ROASTED BEAN EXECUTION:\n---\nSTDOUT:\n{}\n", out_str);
            if !success || !err_str.is_empty() {
                final_out.push_str(&format!(
                    "---\nSTDERR (or cargo compilation logs):\n{}\n",
                    err_str
                ));
            }
            final_out.push_str(&format!(
                "---\nBean saved permanently in {} and recorded in Memento.",
                project_dir
            ));

            ToolResult {
                name: call.name.clone(),
                success,
                output: final_out,
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Cargo execution failed: {}", e),
        },
    }
}

fn execute_python_bean(
    call: &ToolCall,
    bean_name: &str,
    beans_dir: &str,
    code: &str,
    packages: &[String],
) -> ToolResult {
    // Sanitize LLM-generated code: strip single spurious leading spaces.
    // Local models occasionally emit lines with exactly 1 leading space on
    // what should be top-level statements (e.g. " y = np.sin(x)").
    // Real Python indentation uses 2+ spaces; 1-space is almost always an LLM error.
    let sanitized: String = code
        .lines()
        .map(|line| {
            if line.starts_with(' ') && !line.starts_with("  ") {
                line.trim_start_matches(' ')
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let p = format!("{}/{}.py", beans_dir, bean_name);
    if let Err(e) = std::fs::write(&p, &sanitized) {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write Python bean: {}", e),
        };
    }

    let mut pip_log = String::new();
    if !packages.is_empty() {
        let mut cmd = std::process::Command::new("python3");
        cmd.arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--user")
            .arg("--break-system-packages")
            .arg("--quiet");
        for pkg in packages {
            cmd.arg(pkg);
        }
        match cmd.output() {
            Ok(out) if out.status.success() => {
                pip_log = format!("Installed: {:?}", packages);
            }
            Ok(out) => {
                // PEP 668 or already-installed: warn but still run the code.
                // The package may already be available; don't abort.
                let err = String::from_utf8_lossy(&out.stderr);
                pip_log = format!(
                    "pip warning (code still runs): {}",
                    err.lines().next().unwrap_or("non-zero exit")
                );
            }
            Err(e) => {
                pip_log = format!("pip unavailable ({}), proceeding anyway", e);
            }
        }
    }

    match std::process::Command::new("python3").arg(&p).output() {
        Ok(out) => {
            let out_str = String::from_utf8_lossy(&out.stdout).to_string();
            let err_str = String::from_utf8_lossy(&out.stderr).to_string();
            let success = out.status.success();
            let mut res = if success {
                if err_str.trim().is_empty() {
                    out_str
                } else {
                    format!("STDOUT:\n{}\n---\nSTDERR:\n{}", out_str, err_str)
                }
            } else {
                format!("PYTHON SOFT BEAN ERROR:\n{}\n{}", err_str, out_str)
            };
            if !pip_log.is_empty() {
                res = format!("{}\n---\n{}", pip_log, res);
            }
            res = format!(
                "{}\n---\nBean saved permanently at {} and logged in Memento.",
                res, p
            );
            ToolResult {
                name: call.name.clone(),
                success,
                output: res,
            }
        }
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: e.to_string(),
        },
    }
}

pub(crate) async fn execute_write_file(call: &ToolCall) -> ToolResult {
    let path = call
        .arguments
        .get("path")
        .and_then(|p| p.as_str())
        .unwrap_or("");
    let content = call
        .arguments
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");

    if path.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Missing path".into(),
        };
    }

    let resolved_path = match resolve_guarded_fs_path(path, true) {
        Ok(path) => path,
        Err(error) => {
            return ToolResult {
                name: call.name.clone(),
                success: false,
                output: error,
            };
        }
    };

    match std::fs::write(&resolved_path, content) {
        Ok(_) => ToolResult {
            name: call.name.clone(),
            success: true,
            output: format!("Successfully wrote to {}", resolved_path.display()),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Failed to write file: {}", e),
        },
    }
}
