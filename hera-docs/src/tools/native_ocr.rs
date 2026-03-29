use std::collections::HashMap;
use crate::tools::definitions::{ToolArgument, ToolDefinition, build_tool};
use std::process::Command;
use std::fs;
use uuid::Uuid;

pub fn get_native_ocr_tool() -> ToolDefinition {
    build_tool(
        "perform_ocr",
        "Extract text and tabular data natively from an active document (Image or PDF) in your memory. Pass the object_id or memory pointer of the document to extract its full content instantly using deep internal Rust processing.",
        HashMap::from([
            (
                "object_id".to_string(),
                ToolArgument {
                    arg_type: "string".to_string(),
                    description: "The memory ID or pointer of the PDF/Image object to read.".to_string(),
                    enum_values: None,
                },
            ),
            (
                "extract_tables".to_string(),
                ToolArgument {
                    arg_type: "boolean".to_string(),
                    description: "Set to true if prioritizing tabular or pricing data extraction.".to_string(),
                    enum_values: None,
                },
            ),
        ]),
        vec!["object_id".to_string()],
    )
}

pub async fn execute_native_ocr(_object_id: &str, extract_tables: bool, memory_buffer: &[u8]) -> Result<String, String> {
    let is_pdf = memory_buffer.starts_with(b"%PDF");

    if is_pdf {
        match pdf_extract::extract_text_from_mem(memory_buffer) {
            Ok(content) => {
                if !content.trim().is_empty() {
                    return Ok(format!("[Native PDF Extraction Result]\\n\\n{}", content));
                }
            },
            Err(_) => {}
        }
    }

    let temp_id = Uuid::new_v4().to_string();
    let temp_input = format!("/tmp/{}.bin", temp_id);
    let temp_output_base = format!("/tmp/{}", temp_id);
    let temp_output_txt = format!("{}.txt", temp_output_base);

    if let Err(e) = fs::write(&temp_input, memory_buffer) {
        return Err(format!("Failed to write buffer to volatile storage: {}", e));
    }

    let mut cmd = Command::new("tesseract");
    cmd.arg(&temp_input).arg(&temp_output_base).arg("-l").arg("spa+eng");

    if is_pdf {
        if extract_tables {
            let _ = Command::new("pdftotext")
                .arg("-layout")
                .arg(&temp_input)
                .arg(&temp_output_txt)
                .output();
        } else {
            let _ = Command::new("pdftotext")
                .arg(&temp_input)
                .arg(&temp_output_txt)
                .output();
        }
    } else {
        match cmd.output() {
            Ok(output) => {
                if !output.status.success() {
                    let _ = fs::remove_file(&temp_input);
                    return Err(format!("Tesseract failed: {}", String::from_utf8_lossy(&output.stderr)));
                }
            },
            Err(e) => {
                let _ = fs::remove_file(&temp_input);
                return Err(format!("Failed to invoke tesseract: {}", e));
            }
        }
    }

    let result = match fs::read_to_string(&temp_output_txt) {
        Ok(text) => text,
        Err(e) => {
            let _ = fs::remove_file(&temp_input);
            return Err(format!("Failed to read OCR output: {}", e));
        }
    };

    let _ = fs::remove_file(&temp_input);
    let _ = fs::remove_file(&temp_output_txt);

    if result.trim().is_empty() {
        return Err("OCR resulted in empty text.".to_string());
    }

    Ok(format!("[Native OCR Image/Scan Result]\\n\\n{}", result))
}
