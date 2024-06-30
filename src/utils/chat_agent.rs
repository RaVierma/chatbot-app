use langchain_rust::vectorstore::VectorStore;

use langchain_rust::{
    language_models::llm::LLM,
    llm::{OpenAI, OpenAIConfig},
    schemas::{Document, Message},
    similarity_search,
    vectorstore::pgvector::{Store, StoreBuilder},
};

use super::{topic_clasifier::TopicClassifier, vector_space::EmbeddingManager};

pub struct ChatAgent {
    llm: OpenAI<OpenAIConfig>,
    classifier_url: String,
    db_url: String,
    model_name: String,
    embedder_url: String,
    memory: Vec<Message>,
}

impl ChatAgent {
    pub fn new(
        api_base_url: String,
        classifier_url: String,
        api_key: String,
        model_name: String,
        db_url: String,
        embedder_url: String,
    ) -> Self {
        let openconf = OpenAIConfig::new()
            .with_api_base(api_base_url)
            .with_api_key(api_key);
        let llm = OpenAI::new(openconf).with_model(model_name.clone());

        Self {
            llm,
            classifier_url,
            db_url,
            model_name,
            embedder_url,
            memory: vec![Message::new_system_message("
            ### System:
            System: You are a friendly consice assistant that answer the user query using the following pieces of 
            retrieved context to answer the query. If you don't know the answer, or are unsure, say you don't know.
            ")]
        }
    }

    pub async fn get_response(&mut self, query: String) -> String {
        let topic_clasifier = TopicClassifier::new(self.classifier_url.clone());

        let topic = topic_clasifier.classify(query.clone()).await.unwrap();

        let mut ctopic = "other".to_string();
        if topic.to_lowercase().contains("movie") {
            ctopic = "movie".to_string();
        } else if topic.to_lowercase().contains("book") {
            ctopic = "book".to_string();
        }

        let mut _docs = Vec::<Document>::new();

        if ctopic == "book" || ctopic == "movie" {
            let embedding_manager =
                EmbeddingManager::new(&self.model_name, self.embedder_url.clone());

            let col_name = if ctopic == "book" {
                "books_collection"
            } else {
                "movies_collection"
            };
            let store: Store = StoreBuilder::new()
                .embedder(embedding_manager.get_embeddings())
                .collection_name(&col_name)
                .connection_url(&self.db_url)
                .vector_dimensions(2048)
                .build()
                .await
                .unwrap();

            _docs = similarity_search!(store, &topic, 5).await.unwrap();
            if _docs.len() == 0 {
                return "Sorry unable to resolve your query.".to_string();
            }

            let mut prompt_user = "### System: You are a friendly consice assistant that answer the user query 
            using the following pieces of retrieved context to answer the query. If you don't know the answer, or are unsure, 
            say you don't know.\n\n".to_string();
            for d in _docs.iter() {
                let pc = d.page_content.to_string();
                prompt_user.push_str(&pc);
                prompt_user.push_str("\n\n");
            }
            prompt_user.push_str("### User:");
            prompt_user.push_str(&query);
            prompt_user.push_str("### Assistant:");

            let output = self.llm.clone().invoke(&prompt_user).await.unwrap();

            output
        } else {
            self.memory.push(Message::new_human_message(query));
            let response = self
                .llm
                .clone()
                .generate(&self.memory)
                .await
                .map(|res| res.generation)
                .unwrap();

            self.memory.push(Message::new_ai_message(response.clone()));
            response
        }
        // "check".to_string()
    }
}
