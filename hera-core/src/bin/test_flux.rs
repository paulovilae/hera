use anyhow::Result;
use hera_core::ai::engine_flux::FluxEngine;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing engine...");
    let engine = FluxEngine::new()?;
    println!("Generating image...");
    match engine
        .generate_image("A futuristic cybernetic city", 1360, 768)
        .await
    {
        Ok(_) => println!("Success!"),
        Err(e) => println!("Error: {:?}", e),
    }
    Ok(())
}
