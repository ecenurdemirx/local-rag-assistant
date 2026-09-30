# Semantic retriever using cosine similarity over local embeddings
import math
import database
import embeddings
import config

def cosine_similarity(vector1, vector2):
    """Calculates cosine similarity between two float vectors safely."""
    if not vector1 or not vector2:
        return 0.0
    if len(vector1) != len(vector2):
        # Handle dimension mismatch gracefully
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm_1 = math.sqrt(sum(a * a for a in vector1))
    norm_2 = math.sqrt(sum(b * b for b in vector2))

    if norm_1 == 0.0 or norm_2 == 0.0:
        return 0.0

    return dot_product / (norm_1 * norm_2)

def retrieve(query, top_k=config.TOP_K):
    """Generates query embedding and finds the top-K most similar chunks from database."""
    if not query.strip():
        return []

    query_embedding = embeddings.get_embedding(query)

    chunks = database.get_all_chunks()

    scored_chunks = []

    for chunk in chunks:
        stored_emb = chunk.get("embedding")
        if not stored_emb:
            continue

        try:
            score = cosine_similarity(query_embedding, stored_emb)
        except Exception:
            score = 0.0

        scored_chunks.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "score": score
        })

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return scored_chunks[:top_k]

if __name__ == "__main__":
    database.initialize_database()
    question = input("Ask a test question: ")
    results = retrieve(question)
    print(f"\nTop {len(results)} Retrieval Results:")
    for result in results:
        print("\n--------------------")
        print("Source:", result["source"])
        print("Chunk Index:", result["chunk_index"])
        print("Score:", round(result["score"], 4))
        print("Content:", result["content"])
