use crate::ai::{
    ChatMessage, ChatRequest, ChatResponse, ChatStreamResponse, ContentPart, InferenceError,
    LLMEngine, MessageContent,
};
use serde_json::json;
use std::sync::Arc;
use tracing::{info, warn};

/// Intercepts inbound multimodal chat requests to autonomously fetch missing context before generating the final response.
pub struct ContextEngine {
    pub orchestrator: Arc<dyn LLMEngine + Send + Sync>,
    pub main_engine: Arc<dyn LLMEngine + Send + Sync>,
    #[allow(dead_code)]
    pub vision_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
}

impl ContextEngine {
    pub fn new(
        orchestrator: Arc<dyn LLMEngine + Send + Sync>,
        main_engine: Arc<dyn LLMEngine + Send + Sync>,
        vision_engine: Option<Arc<dyn LLMEngine + Send + Sync>>,
    ) -> Self {
        Self {
            orchestrator,
            main_engine,
            vision_engine,
        }
    }

    pub fn vision_engine(&self) -> Option<Arc<dyn LLMEngine + Send + Sync>> {
        self.vision_engine.clone()
    }

    async fn gather_context(&self, req: &ChatRequest) -> Option<String> {
        // Find the last user message
        let last_user_msg = req
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .next_back()?;
        let user_text = match &last_user_msg.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Null => String::new(),
            MessageContent::Parts(parts) => {
                let mut combined = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        combined.push_str(text);
                        combined.push('\n');
                    }
                }
                combined
            }
        };

        if user_text.trim().is_empty() {
            return None;
        }

        // --- LATENCY OPTIMIZATION ---
        // Bypassing the orchestrator pre-computation unless explicitly enabled via env var
        // Saves 5-15 seconds per message for standard conversational queries.
        if std::env::var("HERA_ENABLE_ORCHESTRATOR").unwrap_or_else(|_| "false".to_string())
            != "true"
        {
            return None;
        }
        let system_prompt = r#"You are the Hera Context Orchestrator. The user asked a question. Decide if you need external context to answer it perfectly (e.g. current events, specific facts, recent data). If you need context, use the available tools explicitly by generating a JSON block within a <tool_call> tag. If you have enough info or the user's question doesn't require outside searching, reply EXACTLY with 'NO_CONTEXT_NEEDED'.
        
AVAILABLE TOOLS:
1. 'search_web': Searches the web for recent info. Parameters: {"query": "string"}.
2. 'scrape_url': Reads the page at URL. Parameters: {"url": "string"}.

To invoke a tool, you MUST output exactly this format:
<tool_call>
{"name": "search_web", "arguments": {"query": "the latest news"}}
</tool_call>
"#;

        let mut orch_req = ChatRequest {
            model: "hera-orchestrator".to_string(), // Route through local sovereign engine
            provider: None,
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: MessageContent::Text(system_prompt.to_string()),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(user_text.clone()),
                },
            ],
            tools: Some(vec![
                json!({
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "Searches the web for recent info.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string" }
                            },
                            "required": ["query"]
                        }
                    }
                }),
                json!({
                    "type": "function",
                    "function": {
                        "name": "scrape_url",
                        "description": "Reads the page at URL.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": { "type": "string" }
                            },
                            "required": ["url"]
                        }
                    }
                }),
            ]),
            temperature: Some(0.1),
            max_tokens: Some(512),
            tool_choice: None,
            reasoning_effort: None,
            response_format: None,
            vision_model: None,
            tts_model: None,
            stt_model: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            seed: None,
            stop: None,
            endpoint: None,
            api_key: None,
            stream: None,
            nsfw: None,
        };

        info!("🧠 [ContextEngine] Asking Orchestrator if context is needed...");

        let mut loop_count = 0;
        let mut accumulated_context = String::new();
        let mcp_url =
            std::env::var("HERA_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
        let hera = hera_web::agents::hera::Hera::new(&mcp_url);

        while loop_count < 3 {
            loop_count += 1;

            info!(
                "🧠 [ContextEngine] DEBUG BEFORE EXEC: model='{}', provider='{:?}'",
                orch_req.model, orch_req.provider
            );

            let res = match self.orchestrator.generate_content(orch_req.clone()).await {
                Ok(r) => r,
                Err(e) => {
                    warn!("[ContextEngine] Orchestrator failed: {:?}", e);
                    break;
                }
            };

            if let Some(choice) = res.choices.first() {
                let content = choice.message.content.as_deref().unwrap_or("");

                // If it returned a JSON tool call via Gemini interface
                if let Some(tool_calls) = &choice.message.tool_calls {
                    info!(
                        "🛠️ [ContextEngine] Orchestrator invoked tools natively: {:?}",
                        tool_calls
                    );
                    let mut something_called = false;
                    for tc in tool_calls {
                        let fn_name = tc["function"]["name"].as_str().unwrap_or("");
                        let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                        let args: serde_json::Value =
                            serde_json::from_str(args_str).unwrap_or(json!({}));

                        let tool_result = if fn_name == "search_web" {
                            something_called = true;
                            let query = args["query"].as_str().unwrap_or("");
                            hera.native_web_search(query)
                                .await
                                .unwrap_or_else(|e| format!("Search Error: {}", e))
                        } else if fn_name == "scrape_url" {
                            something_called = true;
                            let url = args["url"].as_str().unwrap_or("");
                            hera.native_web_scrape(url)
                                .await
                                .unwrap_or_else(|e| format!("Scrape Error: {}", e))
                        } else {
                            "Unknown tool".to_string()
                        };

                        accumulated_context
                            .push_str(&format!("Tool '{}' Result:\n{}\n\n", fn_name, tool_result));

                        // Add to history so it knows what it found
                        orch_req.messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: MessageContent::Text(format!("Used {fn_name}.")),
                        });
                        orch_req.messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: MessageContent::Text(format!("Tool Result:\n{}\nDo you need anything else? If no, reply NO_CONTEXT_NEEDED.", tool_result)),
                        });
                    }
                    if something_called {
                        continue;
                    } else {
                        break;
                    }
                }
                // Alternatively, parse text-based <tool_call> injected via NativeEngine templates
                else if content.contains("<tool_call>") {
                    if let Some(start) = content.find("<tool_call>")
                        && let Some(end) = content.find("</tool_call>")
                    {
                        let json_str = &content[start + 11..end];
                        if let Ok(tc) = serde_json::from_str::<serde_json::Value>(json_str) {
                            let fn_name = tc["name"].as_str().unwrap_or("");
                            let args = &tc["arguments"];

                            info!(
                                "🛠️ [ContextEngine] Native Orchestrator invoked tool: {}",
                                fn_name
                            );
                            let tool_result = if fn_name == "search_web" {
                                let query = args["query"].as_str().unwrap_or("");
                                hera.native_web_search(query)
                                    .await
                                    .unwrap_or_else(|e| format!("Search Error: {}", e))
                            } else if fn_name == "scrape_url" {
                                let url = args["url"].as_str().unwrap_or("");
                                hera.native_web_scrape(url)
                                    .await
                                    .unwrap_or_else(|e| format!("Scrape Error: {}", e))
                            } else {
                                "Unknown tool".to_string()
                            };

                            accumulated_context.push_str(&format!(
                                "Tool '{}' Result:\n{}\n\n",
                                fn_name, tool_result
                            ));
                            orch_req.messages.push(ChatMessage {
                                    role: "user".to_string(),
                                    content: MessageContent::Text(format!("Observation from {}:\n{}\nDo you need anything else? If no, reply NO_CONTEXT_NEEDED.", fn_name, tool_result)),
                                });
                            continue;
                        }
                    }
                    break;
                } else if content.trim().contains("NO_CONTEXT_NEEDED") {
                    info!("[ContextEngine] Orchestrator determined no context is needed.");
                    break;
                } else {
                    // It returned some info or thought process without a clear NO_CONTEXT_NEEDED, just capture it and break
                    accumulated_context.push_str(content);
                    break;
                }
            } else {
                break;
            }
        }

        if accumulated_context.is_empty() {
            None
        } else {
            Some(accumulated_context)
        }
    }

    fn inject_context(&self, req: &mut ChatRequest, ctx: &str) {
        if let Some(first) = req.messages.first_mut()
            && first.role == "system"
        {
            match &mut first.content {
                MessageContent::Text(t) => {
                    *t = format!(
                        "{}\n\n[SYSTEM ORCHESTRATOR INJECTED CONTEXT]:\n{}\n\nUse this context to answer the user's latest query if relevant.",
                        t, ctx
                    );
                }
                MessageContent::Parts(parts) => {
                    parts.push(ContentPart::Text {
                            text: format!("\n\n[SYSTEM ORCHESTRATOR INJECTED CONTEXT]:\n{}\n\nUse this context to answer the user's latest query if relevant.", ctx),
                        });
                }
                MessageContent::Null => {
                    first.content = MessageContent::Text(format!(
                        "[SYSTEM ORCHESTRATOR INJECTED CONTEXT]:\n{}\n\nUse this context to answer the user's latest query if relevant.",
                        ctx
                    ));
                }
            }
            return;
        }

        // If no system prompt, insert one
        req.messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(format!("[SYSTEM ORCHESTRATOR INJECTED CONTEXT]:\n{}\n\nUse this context to answer the user's latest query if relevant.", ctx)),
        });
    }
}

#[async_trait::async_trait]
impl LLMEngine for ContextEngine {
    async fn generate_content(&self, mut req: ChatRequest) -> Result<ChatResponse, InferenceError> {
        // Run Context Collector
        if let Some(context) = self.gather_context(&req).await {
            info!("✨ [ContextEngine] Injecting gathered context into main request!");
            self.inject_context(&mut req, &context);
        }

        // Forward to Main Engine
        self.main_engine.generate_content(req).await
    }

    async fn generate_stream(
        &self,
        mut req: ChatRequest,
    ) -> Result<
        tokio::sync::mpsc::Receiver<Result<ChatStreamResponse, InferenceError>>,
        InferenceError,
    > {
        // Run Context Collector
        if let Some(context) = self.gather_context(&req).await {
            info!("✨ [ContextEngine] Injecting gathered context into stream request!");
            self.inject_context(&mut req, &context);
        }

        // Forward to Main Engine
        self.main_engine.generate_stream(req).await
    }
}
