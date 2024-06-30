use langchain_rust::{
    add_documents,
    embedding::ollama::OllamaEmbedder,
    schemas::Document,
    similarity_search,
    vectorstore::{
        pgvector::{Store, StoreBuilder},
        VectorStore,
    },
};
use serde_json::{Result, Value};
use std::{collections::HashMap, fs::OpenOptions, io::BufReader};

#[derive(Debug)]
pub struct EmbeddingManager<'a> {
    pub model_name: &'a str,
    pub api_base_url: String,
}

impl<'a> EmbeddingManager<'a> {
    pub fn new(model_name: &'a str, api_base_url: String) -> Self {
        Self {
            model_name,
            api_base_url,
        }
    }

    pub fn get_embeddings(&self) -> OllamaEmbedder {
        OllamaEmbedder::default()
            .with_api_base(self.api_base_url.to_string())
            .with_model(self.model_name)
    }
}

#[derive(Debug)]
pub struct VectorSpaceManager<'a> {
    pub embedding_manager: EmbeddingManager<'a>,
    pub db_url: String,
    pub collection_name: String,
    pub pre_delete_collection: bool,
}

impl<'a> VectorSpaceManager<'a> {
    pub fn new(
        embedding_manager: EmbeddingManager<'a>,
        db_url: String,
        collection_name: String,
        pre_delete_collection: bool,
    ) -> Self {
        Self {
            embedding_manager,
            db_url,
            collection_name,
            pre_delete_collection,
        }
    }

    pub async fn create_vector_space(&self, documents: Vec<Document>) -> Store {
        let store = StoreBuilder::new()
            .embedder(self.embedding_manager.get_embeddings())
            .pre_delete_collection(self.pre_delete_collection)
            .collection_name(&self.collection_name)
            .connection_url(&self.db_url)
            .vector_dimensions(2048)
            .build()
            .await
            .unwrap();

        let _ = add_documents!(store, &documents).await.map_err(|e| {
            println!("Error adding documents: {:?}", e);
        });
        store
    }
}

pub trait DataLoader {
    fn load_data(&self, json_file_path: String) -> Result<Value> {
        let file = OpenOptions::new().read(true).open(json_file_path).unwrap();
        let reader = BufReader::new(file);
        let data: Value = serde_json::from_reader(reader)?;
        // println!("{}", data);

        Ok(data)
    }

    fn create_documents(&self, json_file_path: String, length: Option<u32>) -> Vec<Document> {
        let data: Value = self.load_data(json_file_path).unwrap();
        let mut document: Vec<Document> = Vec::new();
        let mut count = 0u32;

        for item in data.as_array().unwrap() {
            let obj = item.as_object().unwrap();
            let mut meta_data: HashMap<String, Value> = HashMap::new();

            for (k, v) in obj.into_iter() {
                meta_data.entry(k.to_string()).or_insert(v.clone());
            }

            let item_content = self.get_page_content(item);

            let doc = Document::new(item_content).with_metadata(meta_data.clone());
            document.push(doc);

            if length.is_some() {
                if count == length.unwrap() {
                    break;
                }
            }

            println!("########## current count : {count}");

            count += 1;
        }

        document
    }
    fn get_page_content(&self, item: &Value) -> String;
}

#[derive(Debug)]
pub struct BookDataLoader;

#[derive(Debug)]
pub struct MovieDataLoder;

impl DataLoader for BookDataLoader {
    fn get_page_content(&self, item: &Value) -> String {
        let obj = item.as_object().unwrap();

        let genres = obj["genres"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>();

        format!(
            r#"{} {} {} {} {}"#,
            obj["title"].as_str().unwrap(),
            obj["author"].as_str().unwrap(),
            obj["publication_date"].as_str().unwrap(),
            obj["description"].as_str().unwrap(),
            genres.join(" ")
        )
    }
}

impl DataLoader for MovieDataLoder {
    fn get_page_content(&self, item: &Value) -> String {
        let obj = item.as_object().unwrap();
        let movie_genres_list = obj["movie_genres_list"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>();

        let movie_actor_list = obj["movie_actor_list"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>();

        format!(
            r#"{} {} {} {} {}"#,
            obj["title"].as_str().unwrap(),
            obj["release_date"].as_str().unwrap(),
            obj["summary"].as_str().unwrap(),
            movie_genres_list.join(" "),
            movie_actor_list.join(" ")
        )
    }
}

pub async fn process_data<T: DataLoader>(
    json_file_path: String,
    model_name: &str,
    db_url: String,
    collection_name: String,
    pre_delete_collection: bool,
    data_loader_class: T,
    length: Option<u32>,
    embedder_url: &str,
) {
    let embedding_manager = EmbeddingManager::new(&model_name, embedder_url.to_string());

    // Initialize the vector space manager with the embedding manager
    let vector_space_manager = VectorSpaceManager::new(
        embedding_manager,
        db_url,
        collection_name,
        pre_delete_collection,
    );

    let data_loader = data_loader_class;
    let documents = data_loader.create_documents(json_file_path, length);

    //Create and save the vector space in db
    let vector_store = vector_space_manager.create_vector_space(documents).await;

    // perform a search for testing
    let query = "Baby Boy";
    let search_results = similarity_search!(vector_store, query, 5).await.unwrap();

    println!("{:?}", search_results);
}
