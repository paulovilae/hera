pub mod definitions;
pub mod native_ocr;
pub mod native_xls;

use definitions::{ToolArgument, ToolDefinition, build_tool};
use std::collections::HashMap;

pub fn get_smartos_tools() -> Vec<ToolDefinition> {
    vec![
        build_tool(
            "smartos_rbac_check",
            "Check the RBAC role and permissions for a user. Returns their tier (admin/premium/guest), allowed actions, and GPU routing info.",
            HashMap::from([(
                "userId".to_string(),
                ToolArgument {
                    arg_type: "string".to_string(),
                    description: "User ID".to_string(),
                    enum_values: None,
                },
            )]),
            vec!["userId".to_string()],
        ),
        build_tool(
            "smartos_gpu_route",
            "Get the optimal GPU endpoint for a user based on their RBAC tier. Returns the port, quality level, and step count for image generation.",
            HashMap::from([
                (
                    "userId".to_string(),
                    ToolArgument {
                        arg_type: "string".to_string(),
                        description: "User ID for tier-based routing".to_string(),
                        enum_values: None,
                    },
                ),
                (
                    "engine".to_string(),
                    ToolArgument {
                        arg_type: "string".to_string(),
                        description: "Optional engine override".to_string(),
                        enum_values: Some(vec![
                            "quality".to_string(),
                            "turbo".to_string(),
                            "instant".to_string(),
                        ]),
                    },
                ),
            ]),
            vec!["userId".to_string()],
        ),
        build_tool(
            "smartos_discover_services",
            "Discover all running Docker containers in the ImagineOS stack. Returns service names, ports, status, and categories.",
            HashMap::from([(
                "category".to_string(),
                ToolArgument {
                    arg_type: "string".to_string(),
                    description: "Optional filter by category".to_string(),
                    enum_values: None,
                },
            )]),
            vec![],
        ),
        native_ocr::get_native_ocr_tool(),
        native_xls::get_native_xls_tool(),
    ]
}
