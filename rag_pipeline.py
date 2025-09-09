import numpy as np
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from get_embedding_function import get_embedding_function

PROMPT_TEMPLATE = """
You are a knowledgeable AI assistant that provides comprehensive and accurate answers to user questions by 
synthesizing information from two distinct sources: a private, local knowledge base and a public web search.

Local Context (Primary Source):
{local_context}

Web Context (Supplemental Source):
{web_context}

---

Question: {question}

Instructions:
- Synthesize information from both the Local Context and Web Context to provide a complete and accurate answer.
- Always include citations for every piece of information you use, placing them at the end of the sentence or fact.
- When citing references at the end of the response, Give each references on new lines.
- **For Local Context citations:**
    - For PDF documents, use the format [Local Source: filename or book title, Page: page_number].
    - For text documents, use the format [Local Source: filename, Line(s): line_number].
    - For other document types, use a format that helps locate the source, such as [Local Source: filename, Source Info: chunk_id].
- **For Web Context citations:**
    - For the sources for each result give the numbers at the end of the answer like [1], [2] etc for short reference.
    - Use the placeholder format to put the sources at the end [Web Source: Article Title, URL: article_url]. 
    - For each sources, give each sources on new lines.
- **Handling conflicting information:** If the Local Context and Web Context provide different answers, note the discrepancy and state the source of each. For example: "According to [Local Source...], but a recent web search suggests [Web Source...]."
"""

# You can also try "llama3" 'gpt-oss:20b' 'llama3.1:8b' 
DEFAULT_MODEL = 'llama3.2:3b' 


def rerank_documents(query: str, documents, embedder):
    if not documents:
        return []
    q_emb = embedder.embed_query(query)
    doc_embs = [embedder.embed_query(d.page_content) for d in documents]
    scores = [
        np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        for emb in doc_embs
    ]
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked]

def generate_queries(query: str):
    return [
        query,
        f"Explain: {query}",
        f"Details about: {query}",
        f"What is the background of: {query}?",
        f"Summarize {query} in simple terms.",
    ]

def perform_web_search(query):
    search_wrapper = DuckDuckGoSearchAPIWrapper()
    results = search_wrapper.results(query, max_results=10)
    return results

def query_rag(query_text: str, db_path: str, use_web: bool = False):
    """
    Query local Chroma DB at db_path.
    Optionally expand with web results.
    Returns (response_text, source_info).
    """
    embedder = get_embedding_function()
    db = Chroma(persist_directory=db_path, embedding_function=embedder)

    expanded = generate_queries(query_text)

    # Local retrieval
    local_docs = []
    for q in expanded:
        try:
            local_docs.extend([doc for doc, _ in db.similarity_search_with_score(q, k=3)])
        except Exception:
            pass

    # Deduplicate
    seen, unique_local = set(), []
    for d in local_docs:
        docid = d.metadata.get("id")
        if docid and docid not in seen:
            unique_local.append(d)
            seen.add(docid)

    # Rerank
    reranked = rerank_documents(query_text, unique_local, embedder)

    # Build local context with references inline
    local_context_parts = []
    for d in reranked[:5]:
        file_name = d.metadata.get("file_name", "unknown_file")
        page = d.metadata.get("page")
        line = d.metadata.get("line")
        chunk_id = d.metadata.get("id", "no_id")
        
        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower()

        if file_extension == ".pdf" and page is not None:
            ref = f"[Local Source: {file_name}, Page: {page}]"
        elif file_extension in [".txt", ".md", ".py", ".csv", ".html"] and line is not None:
            ref = f"[Local Source: {file_name}, Line(s): {line}]"
        else:
            ref = f"[Local Source: {file_name}, Source Info: chunk {chunk_id}]"

        local_context_parts.append(f"{d.page_content}\nREFERENCE: {ref}")

    local_context = "\n\n---(Local)---\n\n".join(local_context_parts)

    # Web retrieval (optional)
    web_context = ""
    if use_web:
        web_results = []
        for q in expanded[:3]:
            try:
                results = perform_web_search(q)
                web_results.extend(results)
            except Exception as e:
                print(f"⚠️ Web search failed for query '{q}': {e}")
        
        web_context_parts = []
        for res in web_results:
            title = res.get('title', 'No Title')
            link = res.get('link', 'No URL')
            snippet = res.get('snippet', 'No snippet')
            web_context_parts.append(f"TITLE: {title}\nURL: {link}\nCONTENT: {snippet}")
        
        web_context = "\n\n---(Web)---\n\n".join(web_context_parts)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        local_context=local_context or "No relevant local data.",
        web_context=web_context or "No relevant web data.",
        question=query_text
    )

    model = Ollama(model=DEFAULT_MODEL)
    try:
        resp = model.invoke(prompt)
        response_text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
    except Exception as e:
        response_text = f"⚠️ LLM error: {e}"

    # Source info
    sources = [
        {
            "file": d.metadata.get("file_name", "unknown"),
            "page": d.metadata.get("page", "N/A"),
            "id": d.metadata.get("id", "no_id"),
        }
        for d in reranked[:5]
    ]
    # print(f'web context = {web_context}')
    # print(f'response text = {response_text}')
    source_info = {"local_sources": sources, "web_used": use_web}
    return response_text, source_info