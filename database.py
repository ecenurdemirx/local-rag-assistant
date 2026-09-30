# SQLite database manager for local RAG chunks and embeddings
import sqlite3
import json
import config

def initialize_database():
    """Creates the chunks table if it does not already exist."""
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()

def replace_chunks(chunks):
    """Atomically replace the index, preserving the old index if insertion fails."""
    rows = [
        (item["source"], item["chunk_index"], item["content"], json.dumps(item["embedding"]))
        for item in chunks
    ]
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM chunks")
        conn.executemany(
            """INSERT INTO chunks (source, chunk_index, content, embedding)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        conn.execute(
            """INSERT INTO metadata (key, value) VALUES ('embedding_model', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (config.EMBEDDING_MODEL,),
        )

def get_metadata(key):
    """Return one index metadata value, or None for legacy indexes."""
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None

def index_uses_current_model():
    return get_metadata("embedding_model") == config.EMBEDDING_MODEL

def get_all_chunks():
    """Retrieves all chunks along with their stored embedding vectors."""
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT id, source, chunk_index, content, embedding
                          FROM chunks ORDER BY source, chunk_index""")
        rows = cursor.fetchall()
    
    # Return formatted dictionaries with parsed embeddings
    chunks = []
    for row in rows:
        try:
            embedding_list = json.loads(row[4])
        except Exception:
            embedding_list = []
            
        chunks.append({
            "id": row[0],
            "source": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "embedding": embedding_list
        })
    return chunks

def count_chunks():
    """Returns the total number of chunks currently stored in the database."""
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
    return count

if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")
    print("Total chunks:", count_chunks())
