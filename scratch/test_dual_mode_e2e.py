import os
import sys
import json
import asyncio
import httpx

backend_dir = r"b:\Sadhana\NLP\backend"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app, state

# Load cached Bhagavad Gita or Midnight Library
cache_file = r"b:\Sadhana\NLP\backend\cache\books\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b.json"
vector_file = r"b:\Sadhana\NLP\backend\cache\vectors\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b_vectors.npz"

with open(cache_file, "r", encoding="utf-8") as f:
    bdata = json.load(f)

state.current_book_title = bdata["book_title"]
state.current_book_text = bdata["cleaned_text"]
state.characters = bdata["characters"]
state.rag.load_index_from_disk(vector_file, bdata["chunks"])

print(f"Loaded Book: '{state.current_book_title}' with {len(state.characters)} characters.")

async def run_tests():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60.0) as client:
        # TEST 1: Status endpoint
        status_res = await client.get("/api/status")
        assert status_res.status_code == 200
        print("[PASS] /api/status OK:", status_res.json()["active_book"])

        # TEST 2: Classical NLP Mode (Greeting)
        res_greet = await client.post("/api/chat", json={
            "character_name": "Mrs. Elm",
            "message": "Hello Mrs. Elm, lovely to meet you!",
            "engine": "classical"
        })
        assert res_greet.status_code == 200
        data_greet = res_greet.json()
        assert data_greet["engine"] == "classical_nlp"
        assert data_greet["engine_details"]["intent"] == "greeting"
        print("[PASS] Classical Greeting:", data_greet["response"][:80])

        # TEST 3: Classical NLP Mode (Identity)
        res_ident = await client.post("/api/chat", json={
            "character_name": "Mrs. Elm",
            "message": "Who are you and what is your role in this story?",
            "engine": "classical"
        })
        assert res_ident.status_code == 200
        data_ident = res_ident.json()
        assert data_ident["engine"] == "classical_nlp"
        assert data_ident["engine_details"]["intent"] == "identity_inquiry"
        print("[PASS] Classical Identity:", data_ident["response"][:80])

        # TEST 4: Classical NLP Mode (Topical Retrieval & Quote Extraction)
        res_topic = await client.post("/api/chat", json={
            "character_name": "Mrs. Elm",
            "message": "Tell me about the books and regret in the library",
            "engine": "classical"
        })
        assert res_topic.status_code == 200
        data_topic = res_topic.json()
        assert data_topic["engine"] == "classical_nlp"
        assert data_topic["engine_details"]["intent"] == "topical_information_retrieval"
        assert "matched_quote" in data_topic["engine_details"]
        print("[PASS] Classical Topical Quote:", data_topic["engine_details"]["matched_quote"][:80])

        # TEST 5: Classical NLP Mode (Out of domain rejection)
        res_ood = await client.post("/api/chat", json={
            "character_name": "Mrs. Elm",
            "message": "What is quantum computing?",
            "engine": "classical"
        })
        assert res_ood.status_code == 200
        data_ood = res_ood.json()
        assert data_ood["engine_details"]["intent"] == "out_of_domain"
        print("[PASS] Classical Out-Of-Domain Rejection:", data_ood["response"][:90])

        # TEST 6: Compare Endpoint
        res_comp = await client.post("/api/compare", json={
            "character_name": "Mrs. Elm",
            "message": "What should one do when life feels like a disappointment?"
        })
        assert res_comp.status_code == 200
        data_comp = res_comp.json()
        assert "classical" in data_comp
        assert "llm" in data_comp
        print("[PASS] /api/compare Endpoint OK:")
        print("  - Classical Method:", data_comp["classical"]["engine_details"]["method"])
        print("  - LLM Method:", data_comp["llm"]["engine_details"]["method"])
        print("  - Classical Response:\n   ", data_comp["classical"]["response"][:120], "...")
        print("  - LLM Response:\n   ", data_comp["llm"]["response"][:120], "...")

        print("\nALL DUAL MODE INTEGRATION TESTS PASSED SUCCESSFULLY!")

asyncio.run(run_tests())
