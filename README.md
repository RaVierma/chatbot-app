# ChatBot CLI Application
This is a command-line interface (CLI) application for a RAG based ChatBot implemented using LangChain and Rust programming language which recommend movies or books based on user query.

## Features

- **Conversation**: Engage in natural language conversations with the chatbot.
- **Question-Answer**: Ask questions on various topics and receive informative responses.
- **Customization**: Easily extend or customize the chatbot's capabilities using the LangChain library

## Installation

Before installing the ChatBot CLI, ensure you have the following prerequisites:

1. **Rust**: Make sure you have Rust installed. You can install it via [rustup](https://www.rust-lang.org/tools/install).

2. **PostgreSQL**: Install PostgreSQL as the Vector DB for storing chatbot data. You can download and install PostgreSQL from the [official website](https://www.postgresql.org/download/).

After installing PostgreSQL, you need to create a database for the chatbot. You can do this by following these steps:


- Open a terminal and log in to PostgreSQL using the `psql` command:
  
  ```bash
  psql -U postgres

  Once logged in, create a new database for the chatbot. Replace <database_name> with your desired database name:
    CREATE DATABASE <database_name>;

  Optionally, you can create a new user with privileges on the database. Replace <username> and <password> with your desired username and password:
    CREATE USER <username> WITH PASSWORD '<password>';
    GRANT ALL PRIVILEGES ON DATABASE <database_name> TO <username>;
3. **Ollam**: Also you need to Run [Ollam](https://ollama.com) if you want to run LLM model local. You can use tinyllama, neural-chat etc.

Now you're ready to install the ChatBot CLI. Clone this repository and build the application using Cargo:
```bash
    git clone https://github.com/RaVierma/chatbot-cli.git
    cd chatbot-cli
    cargo build --release
