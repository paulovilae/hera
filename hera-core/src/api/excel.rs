use axum::{
    extract::Multipart,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use calamine::{Reader, open_workbook_from_rs, Xlsx};
use serde::Serialize;
use std::io::Cursor;

#[derive(Serialize)]
pub struct ExcelParseResponse {
    pub success: bool,
    pub sheets: Vec<SheetData>,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct SheetData {
    pub name: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

pub async fn parse_excel_upload(mut multipart: Multipart) -> impl IntoResponse {
    let mut file_bytes = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some(file_name) = field.file_name() {
            if file_name.ends_with(".xlsx") || file_name.ends_with(".xls") {
                if let Ok(data) = field.bytes().await {
                    file_bytes = data.to_vec();
                    break;
                }
            }
        }
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ExcelParseResponse {
                success: false,
                sheets: vec![],
                error: Some("No valid Excel file found in upload".to_string()),
            }),
        );
    }

    let cursor = Cursor::new(file_bytes);
    let mut workbook: Xlsx<_> = match open_workbook_from_rs(cursor) {
        Ok(wb) => wb,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ExcelParseResponse {
                    success: false,
                    sheets: vec![],
                    error: Some(format!("Failed to parse Excel file: {}", e)),
                }),
            );
        }
    };

    let mut sheets = Vec::new();

    let sheet_names = workbook.sheet_names().to_owned();
    for sheet_name in sheet_names {
        if let Ok(range) = workbook.worksheet_range(&sheet_name) {
            let mut headers = Vec::new();
            let mut rows = Vec::new();

            for (i, row) in range.rows().enumerate() {
                if i == 0 {
                    // First row as headers
                    headers = row.iter().map(|c| c.to_string()).collect();
                } else {
                    let r = row.iter().map(|c| c.to_string()).collect();
                    rows.push(r);
                }
            }

            sheets.push(SheetData {
                name: sheet_name,
                headers,
                rows,
            });
        }
    }

    (
        StatusCode::OK,
        Json(ExcelParseResponse {
            success: true,
            sheets,
            error: None,
        }),
    )
}
