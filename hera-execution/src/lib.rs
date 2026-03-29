pub mod bridge;
pub mod tools;
#[cfg(feature = "vector")]
pub mod memory;
pub mod agents;
pub mod workflow;
pub mod dify;

pub fn init() {
    println!("Execution Layer initialized.");
}
