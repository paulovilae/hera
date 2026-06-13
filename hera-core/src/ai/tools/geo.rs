//! Sovereign geocoding tool executors (Nominatim self-hosted via `GEOCODER_URL`).
//! Split out of `platform.rs` to keep that file under the 1500-line domain limit.
use crate::ai::tool_executor::{ToolCall, ToolResult};
use serde_json::json;
use tracing::info;

/// Geocodificación soberana (Nominatim self-hosted, http_adapter): dirección -> coordenada + barrio.
/// URL del servicio en `GEOCODER_URL` (default `http://localhost:8090`, el Nominatim de genesis).
pub(crate) async fn execute_geocode(call: &ToolCall) -> ToolResult {
    let address = call
        .arguments
        .get("address")
        .and_then(|a| a.as_str())
        .unwrap_or("")
        .trim();
    if address.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Falta el parámetro 'address'.".into(),
        };
    }
    let city = call
        .arguments
        .get("city")
        .and_then(|c| c.as_str())
        .unwrap_or("Cali");
    let base = std::env::var("GEOCODER_URL").unwrap_or_else(|_| "http://localhost:8090".to_string());

    // Cascada de respaldo para la nomenclatura colombiana (el número de casa rara
    // vez está en OSM): dirección completa -> vía + barrio -> barrio -> vía.
    let barrio_hint = address
        .to_lowercase()
        .find("barrio ")
        .map(|i| {
            address[i + "barrio ".len()..]
                .split([',', '.'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty());
    let via = address
        .split(['#', ','])
        .next()
        .unwrap_or(address)
        .split(" No.")
        .next()
        .unwrap_or(address)
        .trim();
    let mut attempts = vec![format!("{address}, {city}, Colombia")];
    if let Some(b) = &barrio_hint {
        attempts.push(format!("{via}, {b}, {city}, Colombia"));
        attempts.push(format!("{b}, {city}, Colombia"));
    }
    attempts.push(format!("{via}, {city}, Colombia"));

    let client = reqwest::Client::new();
    for q in &attempts {
        if let Some(first) = nominatim_first(&client, &base, q).await {
            let lat = first.get("lat").and_then(|v| v.as_str()).unwrap_or("");
            let lon = first.get("lon").and_then(|v| v.as_str()).unwrap_or("");
            let name = first.get("display_name").and_then(|v| v.as_str()).unwrap_or("");
            let barrio = first
                .get("address")
                .and_then(|a| {
                    a.get("neighbourhood")
                        .or_else(|| a.get("suburb"))
                        .or_else(|| a.get("city_district"))
                })
                .and_then(|v| v.as_str())
                .unwrap_or("");
            info!("🗺️ [Hera] Geocoded: {address}");
            return ToolResult {
                name: call.name.clone(),
                success: true,
                output: json!({"lat": lat, "lng": lon, "neighbourhood": barrio, "display_name": name})
                    .to_string(),
            };
        }
    }
    ToolResult {
        name: call.name.clone(),
        success: false,
        output: format!("Sin resultados para la dirección '{address}' en {city}."),
    }
}

/// Primer resultado del `/search` de Nominatim para una query libre, o None.
async fn nominatim_first(
    client: &reqwest::Client,
    base: &str,
    q: &str,
) -> Option<serde_json::Value> {
    let url = format!("{}/search", base.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .query(&[
            ("format", "jsonv2"),
            ("limit", "1"),
            ("countrycodes", "co"),
            ("addressdetails", "1"),
            ("q", q),
        ])
        .send()
        .await
        .ok()?;
    let items: serde_json::Value = resp.json().await.ok()?;
    items.as_array()?.first().cloned()
}

/// Reverse geocoding soberano (http_adapter): coordenada -> barrio/lugar.
pub(crate) async fn execute_reverse_geocode(call: &ToolCall) -> ToolResult {
    let getnum = |k: &str| {
        call.arguments.get(k).and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_f64().map(|f| f.to_string()))
        })
    };
    let lat = getnum("lat").unwrap_or_default();
    let lon = getnum("lng").or_else(|| getnum("lon")).unwrap_or_default();
    if lat.is_empty() || lon.is_empty() {
        return ToolResult {
            name: call.name.clone(),
            success: false,
            output: "Faltan los parámetros 'lat' y/o 'lng'.".into(),
        };
    }
    let base = std::env::var("GEOCODER_URL").unwrap_or_else(|_| "http://localhost:8090".to_string());
    let url = format!("{}/reverse", base.trim_end_matches('/'));
    let client = reqwest::Client::new();
    match client
        .get(&url)
        .query(&[
            ("format", "jsonv2"),
            ("addressdetails", "1"),
            ("lat", lat.as_str()),
            ("lon", lon.as_str()),
        ])
        .send()
        .await
    {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(j) => {
                let name = j.get("display_name").and_then(|v| v.as_str()).unwrap_or("");
                let barrio = j
                    .get("address")
                    .and_then(|a| {
                        a.get("neighbourhood")
                            .or_else(|| a.get("suburb"))
                            .or_else(|| a.get("city_district"))
                    })
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if name.is_empty() {
                    ToolResult {
                        name: call.name.clone(),
                        success: false,
                        output: format!("Sin lugar en ({lat},{lon})."),
                    }
                } else {
                    info!("🗺️ [Hera] Reverse-geocoded: {lat},{lon}");
                    ToolResult {
                        name: call.name.clone(),
                        success: true,
                        output: json!({"neighbourhood": barrio, "display_name": name}).to_string(),
                    }
                }
            }
            Err(e) => ToolResult {
                name: call.name.clone(),
                success: false,
                output: format!("Reverse parse error: {e}"),
            },
        },
        Err(e) => ToolResult {
            name: call.name.clone(),
            success: false,
            output: format!("Reverse request failed: {e}"),
        },
    }
}
