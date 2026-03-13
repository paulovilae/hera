use std::collections::HashMap;
use crate::tools::definitions::{ToolArgument, ToolDefinition, build_tool};
use calamine::{open_workbook_auto, Reader, Error, Data};
use std::fs;
use uuid::Uuid;

/// Returns the ToolDefinition for the internal native XLS/XLSX tool.
pub fn get_native_xls_tool() -> ToolDefinition {
    build_tool(
        "extract_xls",
        "Extract tabular data natively from an active spreadsheet document (.xls, .xlsx, .ods) in your memory. Pass the object_id or memory pointer of the document to extract its full content as CSV/Markdown text instantly.",
        HashMap::from([
            (
                "object_id".to_string(),
                ToolArgument {
                    arg_type: "string".to_string(),
                    description: "The memory ID or pointer of the Spreadsheet object to read.".to_string(),
                    enum_values: None,
                },
            )
        ]),
        vec!["object_id".to_string()],
    )
}

/// Executes the native XLS extraction.
/// Reads the byte buffer, writes to a temporary file, parses with calamine, and returns text.
pub async fn execute_native_xls(_object_id: &str, memory_buffer: &[u8]) -> Result<String, String> {
    
    // Calamine needs a file or a Seekable reader. Since `memory_buffer` is just bytes,
    // the easiest way for auto-detection of format is to dump it to a fast tmpfs file.
    
    let temp_id = Uuid::new_v4().to_string();
    let temp_input = format!("/tmp/{}.spreadsheet", temp_id);

    // Write binary from memory to a temp file
    if let Err(e) = fs::write(&temp_input, memory_buffer) {
        return Err(format!("Failed to write buffer to volatile storage: {}", e));
    }

    // Process with Calamine
    let mut excel = match open_workbook_auto(&temp_input) {
        Ok(wb) => wb,
        Err(e) => {
            let _ = fs::remove_file(&temp_input);
            return Err(format!("Failed to open spreadsheet (might be corrupted or unsupported format): {}", e));
        }
    };

    let mut output = String::new();
    let sheet_names = excel.sheet_names().to_owned();
    
    for s_name in sheet_names {
        output.push_str(&format!("\\n## Sheet: {}\\n\\n", s_name));
        
        if let Ok(range) = excel.worksheet_range(&s_name) {
            for row in range.rows() {
                let row_str: Vec<String> = row.iter().map(|c| {
                    match c {
                        Data::Int(n) => n.to_string(),
                        Data::Float(f) => f.to_string(),
                        Data::String(s) => s.replace('\n', " ").replace('\r', ""),
                        Data::Bool(b) => b.to_string(),
                        Data::DateTime(d) => d.to_string(),
                        Data::DateTimeIso(d) => d.to_string(),
                        Data::DurationIso(d) => d.to_string(),
                        Data::Error(e) => format!("Error: {}", e),
                        Data::Empty => String::new(),
                    }
                }).collect();
                output.push_str(&row_str.join(" | "));
                output.push('\n');
            }
        }
    }

    // Cleanup volatile memory files
    let _ = fs::remove_file(&temp_input);

    if output.trim().is_empty() {
         return Err("Spreadsheet resulted in empty tabular text.".to_string());
    }

    Ok(format!("[Native XLS/Spreadsheet Extraction Result]\\n\\n{}", output.trim()))
}
