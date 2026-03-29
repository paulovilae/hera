#![cfg(feature = "vector")]

use anyhow::{Context, Result};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::connection::Connection;
use arrow_array::{RecordBatch, StringArray, RecordBatchIterator, Array};
use arrow_schema::{Field, Schema, DataType};
use std::sync::Arc;
use uuid::Uuid;

/// LanceStore encapsulates the embedded Rust LanceDB engine for ultra-fast, 
/// zero-dependency local semantic vector memory.
#[derive(Clone)]
pub struct LanceStore {
    db: Connection,
    table_name: String,
}

impl LanceStore {
    /// Initializes an embedded LanceDB connection at the target filesystem path.
    pub async fn new(uri: &str, table_name: &str) -> Result<Self> {
        let db = lancedb::connect(uri).execute().await
            .context("Failed to mount embedded LanceDB connection")?;
            
        Ok(Self {
            db,
            table_name: table_name.to_string(),
        })
    }

    /// Stores raw vectors and their associated payload natively.
    pub async fn store(&self, vectors: Vec<f32>, payload: serde_json::Value) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                 vectors.len() as i32,
            ), false),
            Field::new("payload", DataType::Utf8, true),
        ]));

        let id = Uuid::new_v4().to_string();
        
        let id_array = Arc::new(StringArray::from(vec![id])) as Arc<dyn Array>;
        
        // Wrap flat float array into FixedSizeListArray using `FixedSizeListArray::try_new_from_values` equivalent
        // Arrow 58 removed try_new_from_values, so we must construct it with logical bounds.
        // We will build it via list builder instead.
        let mut list_builder = arrow_array::builder::FixedSizeListBuilder::new(
            arrow_array::builder::Float32Builder::new(),
            vectors.len() as i32,
        );
        for &v in &vectors {
            list_builder.values().append_value(v);
        }
        list_builder.append(true);
        let list_array = Arc::new(list_builder.finish()) as Arc<dyn Array>;

        let payload_str = payload.to_string();
        let payload_array = Arc::new(StringArray::from(vec![payload_str])) as Arc<dyn Array>;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array, list_array, payload_array],
        )?;

        // Vectorize batch natively for Database insertion
        let schema_ref = schema.clone();
        // LanceDB 0.26 allows generic iterators over Results
        let batches = vec![Ok(batch.clone())];
        let batches_append = vec![Ok(batch)];
        let batch_iter = RecordBatchIterator::new(batches, schema_ref.clone());

        // Ensure table exists; if not, create it natively
        let table_names = self.db.table_names().execute().await?;
        let table = if table_names.contains(&self.table_name) {
            self.db.open_table(&self.table_name).execute().await?
        } else {
            self.db.create_table(&self.table_name, Box::new(batch_iter) as Box<dyn arrow_array::RecordBatchReader + Send>).execute().await?
        };

        // If table existed, append the new batch
        if table_names.contains(&self.table_name) {
            let append_iter = RecordBatchIterator::new(batches_append, schema_ref.clone());
            table.add(Box::new(append_iter) as Box<dyn arrow_array::RecordBatchReader + Send>).execute().await?;
        }

        Ok(())
    }

    /// Searches the embedded table natively using semantic distance.
    pub async fn search(&self, vector: Vec<f32>, limit: u64) -> Result<Vec<serde_json::Value>> {
        let table = self.db.open_table(&self.table_name).execute().await
            .context("Failed to open LanceDB memory table")?;

        let stream = table.query()
            .nearest_to(vector)?
            .limit(limit as usize)
            .execute()
            .await?;

        // Read stream batches and extract payloads
        use futures::StreamExt;
        let mut results = Vec::new();
        let mut stream = stream;

        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let payload_col = batch.column_by_name("payload").unwrap();
            let string_array = payload_col.as_any().downcast_ref::<StringArray>().unwrap();
            
            for i in 0..string_array.len() {
                if string_array.is_valid(i) {
                    let val: serde_json::Value = serde_json::from_str(string_array.value(i))?;
                    results.push(val);
                }
            }
        }

        Ok(results)
    }
}
