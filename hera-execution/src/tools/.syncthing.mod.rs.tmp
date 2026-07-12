pub mod definitions;
#[cfg(feature = "docs")]
pub use hera_docs::tools::native_ocr;
#[cfg(feature = "docs")]
pub use hera_docs::tools::native_xls;

use definitions::{ToolArgument, ToolDefinition, build_tool};
use std::collections::HashMap;

#[cfg(feature = "docs")]
fn map_doc_tool(tool: hera_docs::tools::definitions::ToolDefinition) -> ToolDefinition {
    ToolDefinition {
        name: tool.name,
        description: tool.description,
        input_schema: definitions::InputSchema {
            schema_type: tool.input_schema.schema_type,
            properties: tool
                .input_schema
                .properties
                .into_iter()
                .map(|(key, arg)| {
                    (
                        key,
                        ToolArgument {
                            arg_type: arg.arg_type,
                            description: arg.description,
                            enum_values: arg.enum_values,
                        },
                    )
                })
                .collect(),
            required: tool.input_schema.required,
        },
    }
}

pub fn get_smartos_tools() -> Vec<ToolDefinition> {
    #[allow(unused_mut)]
    let mut tools = vec![
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
    ];

    #[cfg(feature = "docs")]
    {
        tools.push(map_doc_tool(native_ocr::get_native_ocr_tool()));
        tools.push(map_doc_tool(native_xls::get_native_xls_tool()));
    }

    tools
}
