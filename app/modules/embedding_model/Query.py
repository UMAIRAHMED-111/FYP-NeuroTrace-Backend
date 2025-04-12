import os
import faiss
import pickle
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ======================== CONFIG ========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../embedding_model/Speech To Text Model/tiny.en.pt"))
BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, "../DB"))
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


def query_vector_store(user_id: str, query: str, top_k: int = 1) -> list[dict]:
    """
    Query the user's FAISS vector store for top matching chunks.

    Args:
        user_id (str): UUID of the user.
        query (str): Query string.
        top_k (int): Number of top results to return.

    Returns:
        list of dicts with 'text' and 'timestamp' keys.
    """
    # Validate UUID format
    try:
        uuid.UUID(user_id)
    except ValueError:
        raise ValueError("Invalid UUID format provided for user_id.")

    vector_path = os.path.join(BASE_PATH, "VectorStore", str(user_id))
    index_file = os.path.join(vector_path, "chunk_index.faiss")
    meta_file = os.path.join(vector_path, "chunk_metadata.pkl")

    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"❌ Vector DB files not found for user {user_id}.")

    print("📥 Loading vector index and metadata...")
    index = faiss.read_index(index_file)

    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vec = model.encode([query])[0].reshape(1, -1)

    # Perform FAISS search
    _, indices = index.search(query_vec, top_k)

    # Convert timestamps and fetch results
    results = []
    for i in indices[0]:
        item = metadata[i]
        timestamp = datetime.fromisoformat(item["timestamp"])
        results.append({
            "text": item["text"],
            "timestamp": timestamp
        })

    # Sort by recency
    results.sort(key=lambda r: r["timestamp"], reverse=True)

    return results
