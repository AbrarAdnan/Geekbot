# Returns the embeddings object used for Chroma indexing and retrieval.

from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # Using Ollama local embeddings (nomic-embed-text).
    # If you change to a different embeddings provider, update this file.
    return OllamaEmbeddings(model="nomic-embed-text")