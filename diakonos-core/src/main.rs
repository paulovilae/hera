use tracing::info;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();

    let socket_path = std::env::var("DIAKONOS_SOCKET")
        .unwrap_or_else(|_| "/tmp/diakonos.sock".to_string());
    let modules_path = std::env::var("DIAKONOS_MODULES_CONFIG")
        .unwrap_or_else(|_| {
            "/home/paulo/Programs/apps/OS/Hera/diakonos-core/config/modules.json".to_string()
        });
    let modules = diakonos_core::modules::ModuleRegistry::load(modules_path).await;
    let watcher_registry = modules.clone();

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
        loop {
            interval.tick().await;
            if let Some(enabled) = watcher_registry.reload_if_changed().await {
                tracing::info!("Diakonos modules auto-reloaded: {:?}", enabled);
            }
        }
    });

    info!("Starting Diakonos on {}", socket_path);
    diakonos_core::server::serve(&socket_path, modules).await
}
