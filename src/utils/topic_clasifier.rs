use std::result::Result;

use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Deserialize)]
struct ClassificationResp {
    label: String,
    score: f64,
}

#[derive(Clone)]
pub struct TopicClassifier<'a> {
    // pub llm: OpenAI<OpenAIConfig>,
    pub classifier_url: String,
    pub topic: Vec<&'a str>,
}

impl<'a> TopicClassifier<'a> {
    pub fn new(classifier_url: String) -> Self {
        Self {
            classifier_url,
            topic: vec!["movie", "book", "other"],
        }
    }

    pub async fn classify(&self, query: String) -> Result<String, OLLAMAChatModelError> {
        let default_topic = "other".to_string();
        let client = Client::new();
        let url = Url::parse(&self.classifier_url).unwrap();
        let res = client
            .post(url.clone())
            .json(&json!({
                "message": query,
                "labels": self.topic

            }))
            .send()
            .await
            .unwrap();

        if res.status() != 200 {
            return Err(OLLAMAChatModelError::Exception());
        }

        let data: ClassificationResp = res.json().await.unwrap();
        if data.score > 0.5f64 {
            Ok(data.label.to_string())
        } else {
            Ok(default_topic)
        }
    }
}

#[derive(Debug, Error)]
pub enum OLLAMAChatModelError {
    #[error("Some fatel exception. from ollama server api")]
    Exception(),
}
