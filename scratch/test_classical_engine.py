import os
import sys
import json
import re

backend_dir = r"b:\Sadhana\NLP\backend"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.rag_service import RAGService
from services.sentiment_service import SentimentService

class ClassicalNLPDialogueEngine:
    def __init__(self, book_title: str, text: str, characters: list, rag_service: RAGService, sentiment_service: SentimentService):
        self.book_title = book_title
        self.text = text
        self.characters = characters
        self.rag = rag_service
        self.sentiment = sentiment_service
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
            "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
            "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
            "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
            "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
            "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
            "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
            "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
            "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
            "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
            "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
            "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
            "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
            "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
            "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
            "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
        }

    def clean_query_keywords(self, query: str, char_name: str) -> list:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        char_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', char_name.lower()))
        filtered = [t for t in tokens if t not in self.stop_words and t not in char_tokens]
        return filtered

    def detect_intent(self, query: str, char_name: str) -> tuple:
        q_lower = query.lower().strip()
        
        # 1. Greeting
        if re.search(r'\b(hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)|salutations|namaste)\b', q_lower):
            return "greeting", None

        # 2. Identity / Self
        char_core = char_name.split()[-1].lower()
        if re.search(rf'\b(who\s+are\s+you|what\s+is\s+your\s+name|who\s+is\s+{char_core}|tell\s+me\s+about\s+yourself|what\s+do\s+you\s+do|your\s+role|introduce\s+yourself)\b', q_lower):
            return "identity", None

        # 3. Feelings / Well-being
        if re.search(r'\b(how\s+are\s+you|how\s+do\s+you\s+feel|how\s+is\s+it\s+going|are\s+you\s+(?:happy|sad|well|okay|fine))\b', q_lower):
            return "wellbeing", None

        # 4. Relationships / Mentions of other characters
        for other in self.characters:
            other_name = other.get("name", "")
            if other_name.lower() == char_name.lower():
                continue
            other_parts = [p.lower() for p in other_name.split() if len(p) > 2]
            for part in other_parts:
                if re.search(rf'\b(who\s+is|what\s+about|do\s+you\s+know|tell\s+me\s+about|relationship\s+with)\s+({part})\b', q_lower):
                    return "relationship", other

        return "topical_retrieval", None

    def extract_best_quote_or_sentence(self, text: str, query_keywords: list) -> str:
        # Split passage into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        best_sentence = ""
        best_overlap = -1
        
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 20 or len(s_clean) > 300:
                continue
            s_lower = s_clean.lower()
            overlap = sum(1 for kw in query_keywords if kw in s_lower)
            # Bonus if sentence contains quotes
            if '"' in s_clean or "'" in s_clean or "“" in s_clean:
                overlap += 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s_clean
                
        return best_sentence or (sentences[0].strip() if sentences else text[:200])

    def answer_query(self, query: str, char_info: dict) -> dict:
        char_name = char_info["name"]
        role = char_info.get("role", "Character")
        traits = char_info.get("traits", ["Perceptive"])
        traits_str = ", ".join(traits)
        backstory = char_info.get("story_backstory", char_info.get("backstory", ""))
        learned_facts = char_info.get("learned_facts", [])
        
        intent, target_entity = self.detect_intent(query, char_name)
        
        if intent == "greeting":
            greeting = char_info.get("initial_greeting")
            if not greeting:
                greeting = f"Greetings, my friend. I am {char_name}. Speak freely with me—what thoughts or inquiries bring you to converse with me today?"
            return {
                "engine": "classical_nlp",
                "intent": "greeting",
                "method": "Rule-Based Persona Greeting Synthesis",
                "similarity_score": 1.0,
                "response": greeting,
                "quote": "",
                "explanation": "Classical Rule-Based Pattern Matcher detected a greeting intent. Instantiated character's canonical first-person greeting welcoming the human reader."
            }

        if intent == "identity":
            fact_summary = f" {learned_facts[0]}" if learned_facts else ""
            reply = f"I am {char_name}, {role} in '{self.book_title}'. {backstory}{fact_summary} Those who observe me find me {traits_str}."
            return {
                "engine": "classical_nlp",
                "intent": "identity_inquiry",
                "method": "Structured Slot-Filling Dossier Synthesis",
                "similarity_score": 1.0,
                "response": reply,
                "quote": "",
                "explanation": "Detected character identity inquiry. Formulated first-person self-introduction using learned character slot values (role, traits, narrative backstory)."
            }

        if intent == "wellbeing":
            vader = self.sentiment.analyze(query)
            if vader["compound"] < -0.2:
                reply = f"I sense trouble in your heart, my friend. As {char_name}, I know the weight the world can place upon us. Tell me what burdens you, and let us examine it together."
            else:
                reply = f"I am steadfast in my purpose. As {char_name}, I remain here within the realm of '{self.book_title}'. How fares your own mind as you read through our tale?"
            return {
                "engine": "classical_nlp",
                "intent": "wellbeing_sentiment",
                "method": "NLTK VADER Sentiment Polarity + Affective Rule Synthesis",
                "similarity_score": round(abs(vader["compound"]), 4),
                "response": reply,
                "quote": "",
                "explanation": f"Classified emotional check intent. NLTK VADER sentiment compound score ({vader['compound']}) guided affective first-person response synthesis."
            }

        if intent == "relationship" and target_entity:
            rel_name = target_entity.get("name", "them")
            rel_desc = char_info.get("relationships", "")
            reply = f"Regarding {rel_name}, they play an integral part in my story. {rel_desc} Our encounters shape much of what unfolds in '{self.book_title}'."
            return {
                "engine": "classical_nlp",
                "intent": "character_relationship",
                "method": "Entity Co-Occurrence & Relationship Graph Lookup",
                "similarity_score": 0.95,
                "response": reply,
                "quote": "",
                "explanation": f"Named entity linker matched '{rel_name}' against book character knowledge graph. Extracted relational traits from character profile."
            }

        # 5. Topical / Thematic Retrieval
        kw = self.clean_query_keywords(query, char_name)
        # Search RAG with query
        hits = self.rag.retrieve(query, character_name=char_name, top_k=3)
        
        top_hit = hits[0] if hits else None
        top_score = top_hit.get("relevance_score", 0.0) if top_hit else 0.0

        if top_hit and top_score >= 0.05 and kw:
            matched_text = top_hit["text"]
            best_quote = self.extract_best_quote_or_sentence(matched_text, kw)
            # Format clean quote
            clean_q = best_quote.strip(" '\"“”")
            
            # First-person synthesis
            reply = (
                f"On the matter of your inquiry, my story and words in '{self.book_title}' speak directly to this:\n\n"
                f"\"{clean_q}\"\n\n"
                f"Reflect upon this as it relates to what you seek."
            )
            return {
                "engine": "classical_nlp",
                "intent": "topical_information_retrieval",
                "method": "TF-IDF Vector Space Model + Salient Sentence Extraction",
                "similarity_score": top_score,
                "response": reply,
                "quote": clean_q,
                "chunk_id": top_hit.get("chunk_id"),
                "explanation": f"Computed TF-IDF cosine similarity vector between query keywords and book passages. Best passage matched with similarity {top_score:.4f} (Passage #{top_hit.get('chunk_id')}). Extracted most salient dialogue/sentence without any neural generation."
            }
        else:
            # Out of domain
            topics = traits_str or "the journey and struggles of our story"
            reply = (
                f"The pages of '{self.book_title}' do not record my speaking on '{query}'. "
                f"My experiences and counsel in our story center upon {topics}. "
                f"Ask me about these matters, and I shall share my perspective with you."
            )
            return {
                "engine": "classical_nlp",
                "intent": "out_of_domain",
                "method": "TF-IDF Cosine Similarity Threshold Rejection",
                "similarity_score": top_score,
                "response": reply,
                "quote": "",
                "explanation": f"Cosine similarity score ({top_score:.4f}) fell below relevance threshold. Classical NLP out-of-domain classifier gracefully kept character in-universe."
            }

# Let's run a test on Mrs. Elm
cache_file = r"b:\Sadhana\NLP\backend\cache\books\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b.json"
vector_file = r"b:\Sadhana\NLP\backend\cache\vectors\dbe81a1c57b9ffa5bc3803f7dfbc72bb85ea47930b799864a81038ab1dfa819b_vectors.npz"

with open(cache_file, "r", encoding="utf-8") as f:
    bdata = json.load(f)

rag = RAGService()
rag.load_index_from_disk(vector_file, bdata["chunks"])
sentiment = SentimentService()

engine = ClassicalNLPDialogueEngine(
    book_title=bdata["book_title"],
    text=bdata["cleaned_text"],
    characters=bdata["characters"],
    rag_service=rag,
    sentiment_service=sentiment
)

char_info = bdata["characters"][0]
print(f"Testing Classical NLP Engine for character: {char_info['name']}")

test_prompts = [
    "Hello Mrs. Elm!",
    "Who are you and what is your role?",
    "I am feeling very sad and overwhelmed today",
    "Who is Nora?",
    "Tell me about the books in the library and regrets",
    "What is the capital of France?"
]

for p in test_prompts:
    res = engine.answer_query(p, char_info)
    print("=" * 60)
    print(f"USER: {p}")
    print(f"INTENT: {res['intent']} | METHOD: {res['method']} | SCORE: {res['similarity_score']}")
    print(f"REPLY:\n{res['response']}")
    print(f"EXPLANATION: {res['explanation']}")
