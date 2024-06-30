mod utils;
use colored::*;
use spinners::{Spinner, Spinners};
use std::{
    env::args,
    io::{self, Write},
};

use utils::{
    chat_agent::ChatAgent,
    vector_space::{process_data, BookDataLoader, MovieDataLoder},
};

use tokio::time::{sleep as Tsleep, Duration as TDuration};

// internal api url
const LLM_SERVER_URL: &str = "http://127.0.0.1:3000/v1";
const CLASSIFIER_URL: &str = "http://127.0.0.1:3000/v1/classifier";

// load data and create embeddings
async fn load_data(
    movie_json_file_path: String,
    book_json_file_path: String,
    number_of_movie_data: Option<u32>,
    number_of_book_data: Option<u32>,
    db_url: &str,
    embedder_url: &str,
    model_name: &str,
) {
    process_data(
        movie_json_file_path,
        model_name,
        db_url.to_string(),
        "books_collection".to_string(),
        true,
        BookDataLoader,
        number_of_book_data,
        embedder_url,
    )
    .await;

    process_data(
        book_json_file_path,
        model_name,
        db_url.to_string(),
        "movies_collection".to_string(),
        true,
        MovieDataLoder,
        number_of_movie_data,
        embedder_url,
    )
    .await;
}

#[tokio::main]
async fn main() {
    // run llm api server in newly Spawns asynchronous task
    tokio::spawn(async move {
        utils::llm_server::llm_apiserver().await;
    });

    Tsleep(TDuration::from_secs(3)).await;

    let arg = args().collect::<Vec<String>>();
    assert_eq!(
        arg.len(),
        2,
        "{}",
        "config.toml file path required.".red().bold()
    );

    let config = utils::config_praser::load_config(arg.get(1).unwrap().to_string()).unwrap();

    let load = config.embedding.create_embedding;
    if load {
        load_data(
            config.embedding.movies_data_path,
            config.embedding.books_data_path,
            config.embedding.number_of_movies_data,
            config.embedding.number_of_books_data,
            &config.servers.vector_store_db_url,
            &config.servers.ollama_api_server_url,
            &config.servers.model_name,
        )
        .await;
    }

    println!(
        "==> 🤖 {}: Hello! I'm chatbot How can i help you? 🙂\n",
        "chatbot".red().bold()
    );

    let mut chatagent = ChatAgent::new(
        LLM_SERVER_URL.to_string(),
        CLASSIFIER_URL.to_string(),
        config.servers.api_key.clone(),
        config.servers.model_name.clone(),
        config.servers.vector_store_db_url.clone(),
        config.servers.ollama_api_server_url.clone(),
    );

    loop {
        print!("==> 🧑 {}: ", "You".green().bold());
        io::stdout().flush().unwrap(); // Display prompt to terminal

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();

        let user_input = user_input.trim();

        if user_input == "quit" {
            println!("\n{} 🙂", "Good Bye!!!".blue().bold());
            break;
        }
        let mut sp = Spinner::new(Spinners::Dots9, "".into());

        let output = chatagent
            .get_response(user_input.to_string())
            .await
            .to_string();
        sp.stop();
        println!("==> 🤖 {}: {}\n", "chatbot".red().bold(), output);
    }
}
