//! Construvendo app tool executors — agente Marina (Olave Bay Tower).
//!
//! Clientes DELGADOS de los endpoints JSON del app Construvendo: la lógica y los
//! datos verificados (banco de 176 intents, malla de precios contra el xlsx del
//! cliente) viven en el crate del app (`Apps/Construvendo-rust/src/domain`), no
//! se duplican acá. Esto evita dos fuentes de verdad para el pricing VIS.
use crate::ai::tool_executor::{ToolCall, ToolResult};

/// Base del app Construvendo. Override con `CONSTRUVENDO_URL`; por defecto el
/// puerto local del app (ver `Apps/Construvendo-rust/app.toml`).
fn base_url() -> String {
    std::env::var("CONSTRUVENDO_URL").unwrap_or_else(|_| "http://127.0.0.1:5205".to_string())
}

/// `construvendo_faq` → GET /api/faq?q=... Devuelve la respuesta pre-autorada y
/// validada, o una directiva de derivar a asesor, o pide reformular. El texto
/// nunca lo inventa el modelo: sale del banco del app (seguro para VIS).
pub(crate) async fn execute_construvendo_faq(call: &ToolCall) -> ToolResult {
    let q = call
        .arguments
        .get("q")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if q.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Falta la pregunta del cliente (parámetro q).".into(),
        };
    }

    let url = format!("{}/api/faq", base_url());
    let client = reqwest::Client::new();
    match client.get(&url).query(&[("q", q)]).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(json) => {
                let tipo = json.get("tipo").and_then(|v| v.as_str()).unwrap_or("");
                let output = match tipo {
                    "responder" => json
                        .get("respuesta")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    "derivar_asesor" => "[[SYSTEM DIRECTIVE]]: Este tema requiere validación de un asesor humano (dato jurídico/financiero sensible en un proyecto VIS). NO respondas el dato: dile con amabilidad que quieres darle información exacta y ofrece conectarlo con un asesor. No inventes.".to_string(),
                    _ => "[[SYSTEM DIRECTIVE]]: No hay una respuesta validada para esa pregunta. Pídele al cliente que la reformule u ofrécele agendar una visita a sala de ventas. No inventes datos del proyecto.".to_string(),
                };
                ToolResult {
                    name: call.name.clone(),
                    success: true,
                    output,
                }
            }
            Err(e) => ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("No pude leer la respuesta del proyecto: {e}"),
            },
        },
        Ok(resp) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("El servicio del proyecto respondió con error: {}", resp.status()),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("No pude consultar la información del proyecto ahora mismo: {e}"),
        },
    }
}

/// `construvendo_simular` → GET /api/simular?presupuesto=&mes= Devuelve las
/// unidades que caben en el presupuesto mensual, más accesibles primero.
pub(crate) async fn execute_construvendo_simular(call: &ToolCall) -> ToolResult {
    let presupuesto = call
        .arguments
        .get("presupuesto")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            call.arguments
                .get("presupuesto")
                .and_then(|v| v.as_str())
                .and_then(|s| s.replace(['.', ',', '$', ' '], "").parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    let mes = call
        .arguments
        .get("mes")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    if presupuesto <= 0.0 {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Necesito el presupuesto mensual del cliente en COP para simular.".into(),
        };
    }

    let url = format!("{}/api/simular", base_url());
    let client = reqwest::Client::new();
    match client
        .get(&url)
        .query(&[
            ("presupuesto", presupuesto.to_string()),
            ("mes", mes.to_string()),
        ])
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(json) => ToolResult {
                name: call.name.clone(),
                success: true,
                output: format_simulacion(&json),
            },
            Err(e) => ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("No pude leer la simulación: {e}"),
            },
        },
        Ok(resp) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("El simulador respondió con error: {}", resp.status()),
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("No pude correr la simulación ahora mismo: {e}"),
        },
    }
}

/// Formatea la respuesta del simulador en texto corto y presentable (COP con
/// separador de miles). Recordatorio VIS: es plan de compra de vivienda.
fn format_simulacion(json: &serde_json::Value) -> String {
    let total = json.get("total_disponibles").and_then(|v| v.as_u64()).unwrap_or(0);
    let unidades = json.get("unidades").and_then(|v| v.as_array());
    let Some(unidades) = unidades else {
        return "No pude interpretar la simulación.".to_string();
    };
    if unidades.is_empty() {
        return "Con ese presupuesto mensual no hay unidades dentro del plan de cuota inicial en este momento. Podemos revisar un plazo o una cuota diferente con un asesor.".to_string();
    }
    let mut lineas = Vec::new();
    for u in unidades {
        let numero = u.get("numero").and_then(|v| v.as_u64()).unwrap_or(0);
        let piso = u.get("piso").and_then(|v| v.as_u64()).unwrap_or(0);
        let vista = u.get("vista").and_then(|v| v.as_str()).unwrap_or("");
        let vip = u.get("vip").and_then(|v| v.as_bool()).unwrap_or(false);
        let mensual = u.get("mensual_ref").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let cuota_ini = u.get("cuota_inicial").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let etiqueta_vip = if vip { " (VIP)" } else { "" };
        lineas.push(format!(
            "- Apto {numero} · piso {piso} · vista {vista}{etiqueta_vip}: cuota mensual de referencia {}, cuota inicial {}",
            cop(mensual),
            cop(cuota_ini)
        ));
    }
    format!(
        "Unidades de Olave Bay Tower dentro de ese presupuesto (plan de compra de vivienda VIS, {total} disponibles — muestro las más accesibles):\n{}",
        lineas.join("\n")
    )
}

/// Formatea un monto COP con separador de miles (punto), sin decimales.
fn cop(v: f64) -> String {
    let entero = v.round() as i64;
    let s = entero.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push('.');
        }
        out.push(c);
    }
    let miles: String = out.chars().rev().collect();
    format!("${miles} COP")
}

#[cfg(test)]
mod tests {
    use super::cop;

    #[test]
    fn formatea_cop_con_miles() {
        assert_eq!(cop(1_900_000.0), "$1.900.000 COP");
        assert_eq!(cop(6_840_000.0), "$6.840.000 COP");
        assert_eq!(cop(950.0), "$950 COP");
    }
}
