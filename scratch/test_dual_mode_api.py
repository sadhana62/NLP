import os
import sys
import json
import asyncio

backend_dir = r"b:\Sadhana\NLP\backend"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import state, chat_with_character, compare_both_engines, ChatRequest, CompareRequest

# Load cached Midnight Library into state
cache_file = r"b:\Sadhana\NLP\backend\cache\books\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b.json"
vector_file = r"b:\Sadhana\NLP\backend\cache\vectors\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b_vectors.npz"

with open(cache_file, "r", encoding="utf-8") as f:
    bdata = json.load(f)

state.current_book_title = bdata["book_title"]
state.current_book_text = bdata["cleaned_text"]
state.characters = bdata["characters"]
state.rag.load_index_from_disk(vector_file, bdata["chunks"])

async def run_tests():
    print("=== TEST 1: Classical NLP Mode (Zero LLM) ===")
    req_classical = ChatRequest(
        character_name="Mrs. Elm",
        message="Hello Mrs. Elm, who are you and what do you do?",
        engine="classical"
    )
    res1 = await chat_with_character(req_classical)
    print(f"Engine: {res1.get('engine')}")
    print(f"Response: {res1.get('response')}")
    print(f"Engine Details: {res1.get('engine_details')}")

    print("\n=== TEST 2: Classical NLP Mode - Topical Advice ===")
    req_classical2 = ChatRequest(
        character_name="Mrs. Elm",
        message="Tell me about regrets and the books in the library",
        engine="classical"
    )
    res2 = await chat_with_character(req_classical2)
    print(f"Engine: {res2.get('engine')}")
    print(f"Response:\n{res2.get('response')}")
    print(f"Engine Details: {res2.get('engine_details')}")

    print("\n=== TEST 3: Compare Endpoint ===")
    req_comp = CompareRequest(
        character_name="Mrs. Elm",
        message="What advice do you have when someone feels full of regret?"
    )
    res_comp = await compare_both_engines(req_comp)
    print(f"Query: {res_comp.get('query')}")
    print(f"Classical Reply:\n{res_comp['classical']['response']}")
    print(f"Classical Method: {res_comp['classical']['engine_details']['method']}")
    print(f"LLM Reply:\n{res_comp['llm']['response']}")
    print(f"LLM Method: {res_comp['llm']['engine_details']['method']}")

asyncio.run(run_tests())
