use genpdf::{elements, fonts, Document, SimplePageDecorator};
use serde_json::Value;
use std::io::Cursor;
use tracing::{info, error};

pub fn generate_pdf_from_schema(schema: &Value, template_id: Option<String>) -> Result<Vec<u8>, String> {
    info!("Starting PDF generation in universal pdf_assembler (template: {:?})", template_id);
    
    // Load font from the system
    let font_dir = "/usr/share/fonts/truetype/liberation";
    let font_family = match fonts::from_files(font_dir, "LiberationSans", None) {
        Ok(family) => family,
        Err(e) => {
            error!("Failed to load font from {}: {}", font_dir, e);
            return Err("Failed to load PDF fonts".to_string());
        }
    };

    let mut doc = Document::new(font_family);
    doc.set_title("Hera Universal Document");
    
    let mut decorator = SimplePageDecorator::new();
    decorator.set_margins(15);
    doc.set_page_decorator(decorator);

    // Document Title
    let mut title = elements::Paragraph::new("Hera Generated Document");
    title.set_alignment(genpdf::Alignment::Center);
    doc.push(title);
    doc.push(elements::Break::new(2));

    // Dynamic schema rendering
    if let Value::Object(map) = schema {
        for (k, v) in map {
            let val_str = match v {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => "null".to_string(),
                Value::Array(arr) => format!("{:?}", arr),
                Value::Object(obj) => format!("{:?}", obj),
            };
            doc.push(elements::Paragraph::new(format!("{}: {}", k, val_str)));
            doc.push(elements::Break::new(1));
        }
    } else {
        doc.push(elements::Paragraph::new(schema.to_string()));
    }

    let mut buf = Cursor::new(Vec::new());
    if let Err(e) = doc.render(&mut buf) {
        error!("Failed to render PDF: {}", e);
        return Err("Failed to render PDF".to_string());
    }

    info!("Successfully generated PDF buffer");
    Ok(buf.into_inner())
}
