use std::fs::OpenOptions;
use std::io::{Error, ErrorKind, Read};
use std::path::Path;
use std::result::Result;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub servers: Servers,
    pub embedding: Embedding,
}

#[derive(Debug, Deserialize)]
pub struct Servers {
    // pub llm_server_url: String,
    pub ollama_api_server_url: String,
    // pub classifier_url: String,
    pub api_key: String,
    pub model_name: String,
    pub vector_store_db_url: String,
}

#[derive(Debug, Deserialize)]
pub struct Embedding {
    pub movies_data_path: String,
    pub number_of_movies_data: Option<u32>,
    pub books_data_path: String,
    pub number_of_books_data: Option<u32>,
    pub vector_dimensions: u16,
    pub pre_delete_embeddings: bool,
    pub create_embedding: bool,
}

pub fn load_config(file_path: String) -> Result<Config, Error> {
    let pth = Path::new(&file_path).is_file();
    if pth == false {
        return Err(Error::new(ErrorKind::NotFound, "File not found."));
    }
    let file = OpenOptions::new().read(true).open(file_path);

    let mut config_content = String::default();

    let _ = file.unwrap().read_to_string(&mut config_content);

    let parse_toml = toml::from_str(&config_content);
    if parse_toml.is_err() {
        return Err(Error::new(ErrorKind::NotFound, "Not vaild config file."));
    } else {
        let config_file: Config = parse_toml.unwrap();
        Ok(config_file)
    }
}
