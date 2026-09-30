# Local RAG Pipeline Coordinator
import database
import config
import document_loader
import embeddings
import retriever
import llm

def ingest_documents():
    """Reads local documents, chunks them, generates real embeddings, and updates SQLite."""
    database.initialize_database()
    
    # Build in memory first so a model failure cannot erase the previous index.
    documents = document_loader.load_documents()
    if not documents:
        raise ValueError("No readable documents found; existing index preserved.")
    
    total_chunks = 0
    pending_chunks = []
    
    for doc in documents:
        chunks = document_loader.chunk_text(doc["text"])
        
        vectors = embeddings.get_embeddings(chunks)
        for idx, (chunk_content, embedding_vector) in enumerate(zip(chunks, vectors)):
            pending_chunks.append({
                "source": doc["source"],
                "chunk_index": idx,
                "content": chunk_content,
                "embedding": embedding_vector,
            })
            total_chunks += 1

    if not pending_chunks:
        raise ValueError("No non-empty document chunks found; existing index preserved.")

    database.replace_chunks(pending_chunks)
            
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {total_chunks}")
    print(f"Embeddings generated: {total_chunks}")
    print("Database updated successfully.")
    
    return len(documents), total_chunks

def query_pipeline(question):
    """Executes the retrieve-and-generate RAG pipeline for a user question."""
    # Handle empty/whitespace questions
    if not question.strip():
        return {
            "answer": "Please enter a valid question.",
            "retrieved_chunks": [],
            "sources": []
        }
        
    database.initialize_database()
    
    # If the database is empty, return a helper message
    if database.count_chunks() == 0:
        return {
            "answer": "The knowledge base is currently empty. Please run 'reindex' or run with '--reindex' to ingest documents.",
            "retrieved_chunks": [],
            "sources": []
        }

    if not database.index_uses_current_model():
        return {
            "answer": "The index was built with a different embedding model. Please run with '--reindex'.",
            "retrieved_chunks": [],
            "sources": [],
        }
        
    try:
        retrieved_chunks = retriever.retrieve(question)
    except Exception as error:
        return {
            "answer": f"Local retrieval error: {error}",
            "retrieved_chunks": [],
            "sources": [],
        }
    
    # Filter retrieved chunks by similarity threshold.
    retrieved_chunks = [c for c in retrieved_chunks if c["score"] >= config.SIMILARITY_THRESHOLD]
    
    # If no relevant chunks are found, return the fallback message immediately
    if not retrieved_chunks:
        return {
            "answer": llm.INSUFFICIENT_CONTEXT_RESPONSE,
            "retrieved_chunks": [],
            "sources": []
        }
    
    try:
        answer = llm.generate_answer(question, retrieved_chunks)
    except RuntimeError as error:
        return {
            "answer": f"Local model error: {error}",
            "retrieved_chunks": retrieved_chunks,
            "sources": [],
        }

    
    if answer == llm.INSUFFICIENT_CONTEXT_RESPONSE:
        return {"answer": answer, "retrieved_chunks": retrieved_chunks, "sources": []}

    sources = []
    for chunk in retrieved_chunks:
        if chunk["source"] not in sources:
            sources.append(chunk["source"])
            
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources
    }
