#![allow(dead_code, unused)]
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use axum::{response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};

use reqwest::{Client, Error as ReqwestError, StatusCode, Url};
use serde_json::json;

#[derive(Debug, Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletions {
    messages: Vec<Message>,
    model: String,
    candidate_count: Option<usize>,
    max_tokens: Option<u16>,
    temperature: Option<f32>,
    stop_words: Option<Vec<String>>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<usize>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    n: Option<usize>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct OLLAMAChatModelGenerateResponse {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_duration: u64,
    eval_count: u16,
    eval_duration: u64,
}

#[derive(Debug, Deserialize)]
struct OLLAMAChatModelResponse {
    model: String,
    created_at: String,
    message: Message,
    done: bool,
    context: Option<Vec<u64>>,
    total_duration: u64,
    load_duration: u64,
    // prompt_eval_count: u64,
    prompt_eval_duration: u64,
    eval_count: u16,
    eval_duration: u64,
}

#[derive(Debug, Deserialize)]
struct TopLogProbs {
    token: String,
    logprob: i32,
    bytes: Option<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
struct LogprobsContent {
    token: String,
    logprob: i32,
    bytes: Option<Vec<u8>>,
    top_logprobs: Vec<TopLogProbs>,
}

#[derive(Debug, Deserialize)]
struct Logprobs {
    content: Option<Vec<LogprobsContent>>,
}

#[derive(Debug, Deserialize)]
struct Choices {
    index: u16,
    message: Message,
    logprobs: Option<Logprobs>,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    system_fingerprint: String,
    choices: Vec<Choices>,
    usage: Usage,
}

#[derive(Debug)]
pub struct OLLAMAChatModel {
    model: String,
    base_url: String,
}

impl Default for OLLAMAChatModel {
    fn default() -> Self {
        let model = String::from("tinyllama:chat");
        let base_url = String::from("http://localhost:11434");
        OLLAMAChatModel::new(model, base_url)
    }
}

impl OLLAMAChatModel {
    pub fn new(model: String, base_url: String) -> Self {
        Self { model, base_url }
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        max_tokens: Option<u16>,
    ) -> Result<String, OLLAMAChatModelError> {
        let client = Client::new();
        // let url = Url::parse(&format!("{}{}", self.base_url, "/api/generate")).unwrap();
        let url = Url::parse(&format!("{}{}", self.base_url, "/api/chat")).unwrap();
        let mut promtp_len = 0;
        for msg in messages.iter() {
            promtp_len += msg.content.split(" ").count();
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f32() as u32;
        let value = json!({
            "model": &self.model,
            "messages": messages,
            "stream": false,
            "options": {
                // "top_k": top_k,
                "num_predict": max_tokens,
                "temperature": 0.95
               }
        });

        println!("##################################$$$$ message ${}", value);

        let res = client.post(url.clone()).json(&value).send().await?;

        if res.status() != 200 {
            return Err(OLLAMAChatModelError::HttpError {
                status_code: res.status(),
                error_message: format!("Received non-200 response: {}", res.status()),
            });
        }
        let data: OLLAMAChatModelResponse = res.json().await.unwrap();
        let completion_len = data.message.content.split(" ").count();
        println!("{:?}", data);
        let a = json!({
          "id": format!("chatcmpl-{}", promtp_len),
          "object": "chat.completion",
          "created": now,
          "model": "gpt-3.5-turbo-0125",
          "system_fingerprint": format!("fp_4470{}6fcb", promtp_len),
          "choices": [{
            "index": 0,
            "message": data.message,
            "logprobs": null,
            "finish_reason": "stop"
          }],
          "usage": {
            "prompt_tokens": promtp_len,
            "completion_tokens": completion_len,
            "total_tokens": completion_len + promtp_len
          }
        });

        Ok(a.to_string())
    }

    pub async fn generate(
        &self,
        query: String,
        max_tokens: Option<u16>,
    ) -> Result<String, OLLAMAChatModelError> {
        let client = Client::new();
        let url = Url::parse(&format!("{}{}", self.base_url, "/api/generate")).unwrap();
        let promtp_len = query.len();
        let res = client
            .post(url.clone())
            .json(&json!({
                "prompt": query,
                "model": &self.model,
                "stream": false,
                "temperature": 0.35,
                "raw": true,
                "options": {
                    "num_predict": max_tokens,
                    "top_k": 30,
                    "top_p": 0.3
                   }
            }))
            .send()
            .await?;

        if res.status() != 200 {
            return Err(OLLAMAChatModelError::HttpError {
                status_code: res.status(),
                error_message: format!("Received non-200 response: {}", res.status()),
            });
        }
        let data: OLLAMAChatModelGenerateResponse = res.json().await?;
        println!("### {:?}", data);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f32() as u32;
        let completion_len = data.response.split(" ").count();
        let a = json!({
          "id": format!("chatcmpl-{}", promtp_len),
          "object": "chat.completion",
          "created": now,
          "model": "gpt-3.5-turbo-0125",
          "system_fingerprint": format!("fp_4470{}6fcb", promtp_len),
          "choices": [{
            "index": 0,
            "message": {
                  "role": "assistant",
                  "content":  data.response
              }
            ,
            "logprobs": null,
            "finish_reason": "stop"
          }],
          "usage": {
            "prompt_tokens": promtp_len,
            "completion_tokens": completion_len,
            "total_tokens": completion_len + promtp_len
          }
        });

        Ok(a.to_string())
    }
}

#[derive(Debug, Error)]
pub enum OLLAMAChatModelError {
    #[error("Network request failed: {0}")]
    RequestError(#[from] ReqwestError),

    #[error("HTTP error: {status_code} {error_message}")]
    HttpError {
        status_code: StatusCode,
        error_message: String,
    },
    #[error("Some fatel exception.")]
    Exception(),
}

async fn chat_completions(Json(chat_completions): Json<ChatCompletions>) -> impl IntoResponse {
    let ollma = OLLAMAChatModel::default();
    let msg = chat_completions
        .messages
        .get(0)
        .unwrap()
        .content
        .contains("Use the following pieces of context to answer the questio");
    if msg {
        println!("#########3 generate {:?}", chat_completions);
        let resp = ollma
            .generate(
                chat_completions
                    .messages
                    .get(0)
                    .unwrap()
                    .content
                    .to_string(),
                chat_completions.max_tokens,
            )
            .await
            .unwrap();
        resp.to_string()
    } else {
        println!("#########3 {:?}", chat_completions);
        let resp = ollma
            .invoke(chat_completions.messages, chat_completions.max_tokens)
            .await
            .unwrap();
        resp.to_string()
    }
}

pub async fn llm_apiserver() {
    let app = Router::new().route("/v1/chat/completions", post(chat_completions));

    axum::Server::bind(&"127.0.0.1:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap()
}
