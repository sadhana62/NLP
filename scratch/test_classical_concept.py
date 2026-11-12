import os
import sys
import json
import re

backend_dir = r"b:\Sadhana\NLP\backend"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.rag_service import RAGService
from services.sentiment_service import SentimentService

# Load cached Alice In Wonderland or Midnight Library
cache_file = r"b:\Sadhana\NLP\backend\cache\books\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b.json"
vector_file = r"b:\Sadhana\NLP\backend\cache\vectors\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b_vectors.npz"

with open(cache_file, "r", encoding="utf-8") as f:
    book_data = json.load(f)

rag = RAGService()
rag.load_index_from_disk(vector_file, book_data["chunks"])

print(f"Loaded: {book_data['book_title']} with {len(rag.chunks)} chunks")
print(f"Characters: {[c['name'] for c in book_data['characters']]}")

queries = [
    "Hello Mrs. Elm",
    "Who are you?",
    "Tell me about regrets",
    "Who is Nora?",
    "What is the capital of France?"
]

for q in queries:
    hits = rag.retrieve(q, character_name="Mrs. Elm", top_k=3)
    print(f"\nQuery: '{q}' -> Top Score: {hits[0]['relevance_score'] if hits else 0}")
    if hits:
        print(f"Snippet: {hits[0]['text'][:150]}...")
