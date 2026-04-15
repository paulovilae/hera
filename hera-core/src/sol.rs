//! S.O.L (Semantic Object Language) Parser
//!
//! A standardized grammar for LLMs to query and manipulate the object graph.
//! Grants SQL superpowers against traditional external databases.

#[derive(Debug, Clone, PartialEq)]
pub enum SolOperation {
    Inspect { object_id: String },
    Extract { field: String, object_id: String },
    Join { left_id: String, right_id: String },
    Insert { content: String, object_id: String },
    FindChildren { parent_id: String },
    QuerySystem { database: String, query: String },
}

pub struct SolParser;

impl SolParser {
    /// Extremely fast, non-halting S.O.L parser for extracting graph queries
    /// out of LLM reasoning blocks.
    pub fn parse(input: &str) -> Result<SolOperation, String> {
        let tokens: Vec<&str> = input.split_whitespace().collect();
        if tokens.is_empty() {
            return Err("Empty SOL command".to_string());
        }

        match tokens[0].to_uppercase().as_str() {
            "INSPECT" => {
                if tokens.len() == 2 {
                    Ok(SolOperation::Inspect {
                        object_id: tokens[1].to_string(),
                    })
                } else {
                    Err("Syntax error: INSPECT <object_id>".to_string())
                }
            }
            "EXTRACT" => {
                if tokens.len() == 4 && tokens[2].to_uppercase() == "FROM" {
                    Ok(SolOperation::Extract {
                        field: tokens[1].to_string(),
                        object_id: tokens[3].to_string(),
                    })
                } else {
                    Err("Syntax error: EXTRACT <field> FROM <object_id>".to_string())
                }
            }
            "FIND" => {
                if tokens.len() == 4
                    && tokens[1].to_uppercase() == "CHILDREN"
                    && tokens[2].to_uppercase() == "OF"
                {
                    Ok(SolOperation::FindChildren {
                        parent_id: tokens[3].to_string(),
                    })
                } else {
                    Err("Syntax error: FIND CHILDREN OF <parent_id>".to_string())
                }
            }
            "JOIN" => {
                if tokens.len() == 4 && tokens[2].to_uppercase() == "WITH" {
                    Ok(SolOperation::Join {
                        left_id: tokens[1].to_string(),
                        right_id: tokens[3].to_string(),
                    })
                } else {
                    Err("Syntax error: JOIN <id_1> WITH <id_2>".to_string())
                }
            }
            "INSERT" => {
                if let Some(into_idx) = tokens.iter().rposition(|&t| t.to_uppercase() == "INTO")
                    && into_idx > 1
                    && into_idx + 1 < tokens.len()
                {
                    let content = tokens[1..into_idx].join(" ");
                    let object_id = tokens[into_idx + 1].to_string();
                    return Ok(SolOperation::Insert { content, object_id });
                }
                Err("Syntax error: INSERT <content> INTO <object_id>".to_string())
            }
            "QUERY" => {
                // Format: QUERY SYSTEM <database> WITH [S.O.L->SQL equivalent statements]
                if tokens.len() >= 5 && tokens[1].to_uppercase() == "SYSTEM" {
                    let with_idx = tokens
                        .iter()
                        .position(|&t| t.to_uppercase() == "WITH")
                        .unwrap_or(0);
                    if with_idx > 2 {
                        let database = tokens[2].to_string(); // The system name (vetra_db, latinos_db)
                        let query = tokens[with_idx + 1..].join(" ");
                        return Ok(SolOperation::QuerySystem { database, query });
                    }
                }
                Err("Syntax error: QUERY SYSTEM <db> WITH <query>".to_string())
            }
            _ => Err(format!("Unknown SOL command: {}", tokens[0])),
        }
    }
}

pub fn execute_sol(op: &SolOperation) -> String {
    match op {
        SolOperation::Inspect { object_id } => {
            format!(
                "[S.O.L Execution] INSPECT {}: {{ \"id\": \"{}\", \"type\": \"mock_object\", \"status\": \"inspected\" }}",
                object_id, object_id
            )
        }
        SolOperation::Extract { field, object_id } => {
            format!(
                "[S.O.L Execution] EXTRACT {} FROM {}: \"mock_{}_value\"",
                field, object_id, field
            )
        }
        SolOperation::Join { left_id, right_id } => {
            format!(
                "[S.O.L Execution] JOIN {} WITH {}: {{ \"joined\": true, \"left\": \"{}\", \"right\": \"{}\" }}",
                left_id, right_id, left_id, right_id
            )
        }
        SolOperation::Insert {
            content: _,
            object_id,
        } => {
            format!(
                "[S.O.L Execution] INSERT INTO {}: Successfully inserted content into object.",
                object_id
            )
        }
        SolOperation::FindChildren { parent_id } => {
            format!(
                "[S.O.L Execution] FIND CHILDREN OF {}: [\"{}_child_1\", \"{}_child_2\"]",
                parent_id, parent_id, parent_id
            )
        }
        SolOperation::QuerySystem { database, query } => {
            format!(
                "[S.O.L Execution] QUERY SYSTEM {}: Executed SQL -> \"{}\"",
                database, query
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspect() {
        assert_eq!(
            SolParser::parse("INSPECT obj_contract_123").unwrap(),
            SolOperation::Inspect {
                object_id: "obj_contract_123".to_string()
            }
        );
    }

    #[test]
    fn test_query_system() {
        assert_eq!(
            SolParser::parse("QUERY SYSTEM latinos_db WITH SELECT * FROM trades").unwrap(),
            SolOperation::QuerySystem {
                database: "latinos_db".to_string(),
                query: "SELECT * FROM trades".to_string()
            }
        );
    }
}
