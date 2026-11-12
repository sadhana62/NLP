import re
import numpy as np

# Test the keyword retrieval logic
def test_retrieval(query, char_name, rag):
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    stop_words = {"what", "is", "the", "of", "and", "in", "to", "for", "a", "an", "tell", "me", "about"}
    char_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', char_name.lower()))
    kw = [t for t in tokens if t not in stop_words and t not in char_tokens]
    
    query_str = " ".join(kw)
    print(f"\nOriginal Query: '{query}' -> Informative Keywords: '{query_str}'")
    
    # Retrieve without forcing char_name into the search vector
    hits = rag.retrieve(query_str, character_name="", top_k=3)
    if not hits:
        print("  -> No matches found (Score: 0.0)")
    else:
        print(f"  -> Top Score: {hits[0]['relevance_score']} (Passage #{hits[0]['chunk_id']})")
        print(f"  -> Snippet: {hits[0]['text'][:120]}...")

import sys, json
backend_dir = r"b:\Sadhana\NLP\backend"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.rag_service import RAGService
cache_file = r"b:\Sadhana\NLP\backend\cache\books\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b.json"
vector_file = r"b:\Sadhana\NLP\backend\cache\vectors\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b_vectors.npz"

with open(cache_file, "r", encoding="utf-8") as f:
    bdata = json.load(f)

rag = RAGService()
rag.load_index_from_disk(vector_file, bdata["chunks"])

test_retrieval("Tell me about the books in the library and regrets", "Mrs. Elm", rag)
test_retrieval("What is the capital of France?", "Mrs. Elm", rag)
test_retrieval("I am feeling very sad and overwhelmed today", "Mrs. Elm", rag)
