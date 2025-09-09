# Geekbot: Your Personal AI Document Assistant

This project is a powerful, locally-run AI chatbot designed to answer questions using your private documents. By running entirely on your laptop (16GB RAM or less, no GPU required!), it provides fast, secure, and private access to information from your own files.

This is my starting project dive into the world of LLMs(large language model) to learn about Retrieval-Augmented Generation (RAG), a key technique for building intelligent chatbots that use external data.

![image](https://gist.github.com/user-attachments/assets/2a770415-ce82-41ab-9444-ca3930851070)
#### Features

    100% Local & Private: All processing happens on your machine. Your data and conversations are never sent to a cloud server.

    Document Q&A: Ask questions about your own PDFs, research papers, project notes, or any other supported file.

    Web Augmentation: Optionally perform a live web search on DuckDuckGO to get the latest, most up-to-date information, complementing your local data.

    Multi-Modal RAG: Supports various document types including .pdf, .txt, .md, .csv, and more.

    Friendly UI: An intuitive web interface built with Streamlit makes it easy to chat and manage your documents.

    Efficient: Built to run on a laptop without a dedicated GPU, using quantized models and a highly efficient pipeline.

#### Use Case: Med-Assist AI

While this project is designed for general use, its architecture is perfect for building a specialized tool like a Medical Assistant AI. By indexing medical textbooks, research papers, and clinical guidelines, you can create a secure, private chatbot that provides instant answers for healthcare professionals or students.

🚀 Getting Started

Follow these steps to set up and run the project on your machine.
#### Step 1: Install Ollama

First, download and install Ollama, the platform that runs the language models locally.
[Download Ollama from here](https://ollama.com/download)
#### Step 2: Download the Models

Open your terminal or command prompt and pull the necessary models. 
These are small enough to run on your laptop.

Download the model from the server
`ollama pull qwen3:0.6b`
this is a lightweight model but you can also use other models 
('gpt-oss:20b', 'llama3.1:8b', 'deepseek-r1:1.5b', 'qwen3:0.6b', 'gemma3:270m')
but they may require more processing power

Download the embedding Model (for document understanding)
`ollama pull nomic-embed-text`

#### Step 3: Set up the Project

Clone this repository and install the Python dependencies.

#### Clone the repository
`git clone https://github.com/AbrarAdnan/Geekbot.git`
`cd Geekbot`

#### Create a virtual environment and install dependencies
Python 3.10 is recommended to use for this project
Initialize the virtualenv
`virtualenv venv`
Activate the virtualenv
`source venv/bin/activate  ` For linux
`venv\Scripts\activate` For Windows
Install the dependencies
`pip install -r requirements.txt`

#### Step 4: Run the App

Make sure ollama app is running or run in in the terminal with

`ollama serve`

Launch the Streamlit app to start chatting with your documents.

`streamlit run app.py`

#### Step 5: Enter your files through the streamlit UI
It'll take some time to sync and will show you the progress after it's done

Your browser will automatically open a new tab with the chatbot UI.
💡 Improvements & Future Work

This project can be a solid foundation for more complex applications. Here are some potential improvements to consider for future development:

    Refined Citation: Improve the citation format to be cleaner and more consistent.

    Advanced Web Search: Implement an agentic web search, where the LLM decides if a search is needed and what to search for.

    Multi-modal Input: Add support for image and audio input, allowing the user to upload a chart or a voice note and get answers.

    Model Fine-tuning: Fine-tune a general-purpose LLM on specific domain data to improve its specialized knowledge and response quality.

    UI/UX Enhancements: Add features like chat history saving, model selection within the UI, and a more polished design.

🤝 Project Structure

    app.py: The main Streamlit UI for the chatbot.

    rag_pipeline.py: The core logic for RAG, including retrieval, multi-query expansion, and web search.

    storage_utils.py: Contains shared helper functions for document processing and database management.

    get_embedding_function.py: Defines the embedding model used for converting text into vectors.

    watch.py: A background script that automatically indexes new files as you add them.

    chroma/: The directory where the vector database is stored.

    uploaded/: The files uploaded for indexing will be stored here

Any suggestions for new fuatures and collaboration is welcomed.