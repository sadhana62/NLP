import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("pageecho.classical_nlp")

class ClassicalNLPService:
    """
    Classical NLP Dialogue & Character Synthesis Service (Zero-LLM Mode).
    
    Implements:
    1. Intent Classification via Regex & Semantic Keyword Spotting
    2. Affective Sentiment Analysis via NLTK VADER
    3. Named Entity Linking for Character Relationships
    4. Information Retrieval via TF-IDF Vector Space Model & Cosine Similarity
    5. Salient Dialogue & Quote Extraction from Book Chunks
    6. Rule-Based First-Person Template Response Synthesis
    """

    def __init__(self):
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

    def clean_query_keywords(self, query: str, char_name: str) -> List[str]:
        """Extract substantive search terms from user query, filtering out stop words and character name."""
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        char_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', char_name.lower()))
        return [t for t in tokens if t not in self.stop_words and t not in char_tokens]

    def detect_intent(self, query: str, char_name: str, all_characters: List[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Classify user query into dialogue intent categories using classical pattern heuristics."""
        q_lower = query.lower().strip()

        # 1. Greeting Intent
        if re.search(r'\b(hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)|salutations|namaste)\b', q_lower):
            return "greeting", None

        # 2. Identity / Self-Inquiry Intent
        char_core = char_name.split()[-1].lower()
        if re.search(rf'\b(who\s+are\s+you|what\s+is\s+your\s+name|who\s+is\s+{char_core}|tell\s+me\s+about\s+yourself|what\s+do\s+you\s+do|your\s+role|introduce\s+yourself)\b', q_lower):
            return "identity", None

        # 3. Emotional State / Well-being Intent
        if re.search(r'\b(how\s+are\s+you|how\s+do\s+you\s+feel|how\s+is\s+it\s+going|are\s+you\s+(?:happy|sad|well|okay|fine))\b', q_lower):
            return "wellbeing", None

        # 4. Character Relationship Intent
        if all_characters:
            for other in all_characters:
                other_name = other.get("name", "") if isinstance(other, dict) else str(other)
                if not other_name or other_name.lower() == char_name.lower():
                    continue
                parts = [p.lower() for p in other_name.split() if len(p) > 2]
                for part in parts:
                    if re.search(rf'\b(who\s+is|what\s+about|do\s+you\s+know|tell\s+me\s+about|relationship\s+with)\s+{re.escape(part)}\b', q_lower):
                        return "relationship", other

        return "topical_retrieval", None

    def extract_salient_sentence_or_quote(self, passage_text: str, query_keywords: List[str]) -> str:
        """Find the most informative dialogue quote or sentence inside a retrieved book passage."""
        sentences = re.split(r'(?<=[.!?])\s+', passage_text)
        best_sentence = ""
        best_score = -1

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15 or len(s_clean) > 350:
                continue
            s_lower = s_clean.lower()
            keyword_overlap = sum(1 for kw in query_keywords if kw in s_lower)
            score = keyword_overlap * 2

            # Give priority bonus to dialogue quotes inside the passage
            if '"' in s_clean or "'" in s_clean or "“" in s_clean or "”" in s_clean:
                score += 1.5

            if score > best_score:
                best_score = score
                best_sentence = s_clean

        if best_sentence:
            return best_sentence

        # Fallback to first coherent sentence
        for s in sentences:
            if len(s.strip()) >= 20:
                return s.strip()

        return passage_text[:250].strip()

    def generate_response(
        self,
        query: str,
        char_info: Dict[str, Any],
        book_title: str,
        all_characters: List[Dict[str, Any]],
        rag_service: Any,
        sentiment_service: Any
    ) -> Dict[str, Any]:
        """
        Generate a character dialogue response using pure Classical NLP without any LLM.
        """
        char_name = char_info.get("name", "Character")
        role = char_info.get("role", "Book Figure")
        traits = char_info.get("traits", ["Authentic", "Perceptive"])
        traits_str = ", ".join(traits)
        backstory = char_info.get("story_backstory", char_info.get("backstory", ""))
        learned_facts = char_info.get("learned_facts", [])

        # Step 1: Detect Dialogue Intent
        intent, target_entity = self.detect_intent(query, char_name, all_characters)

        # Step 2: NLTK Sentiment Analysis of User Input
        user_sentiment = sentiment_service.analyze(query)

        # Handle Intent 1: Greeting
        if intent == "greeting":
            greeting = char_info.get("initial_greeting")
            if not greeting:
                greeting = f"Greetings, my friend. I am {char_name}. Speak freely with me—what brings you into the pages of '{book_title}' to converse today?"
            
            char_sentiment = sentiment_service.analyze(greeting)
            return {
                "engine": "classical_nlp",
                "character": char_name,
                "response": greeting,
                "user_sentiment": user_sentiment,
                "character_sentiment": char_sentiment,
                "citations": [],
                "engine_details": {
                    "method": "Rule-Based Persona Greeting Synthesis",
                    "intent": "greeting",
                    "similarity_score": 1.0,
                    "model_used": "None (Pure Classical NLP)",
                    "explanation": "Detected greeting intent via regex token matching. Retrieved character's canonical first-person opening greeting welcoming the human reader."
                }
            }

        # Handle Intent 2: Character Identity
        if intent == "identity":
            fact_lead = f" {learned_facts[0]}" if learned_facts else ""
            reply = (
                f"I am {char_name}, {role} in '{book_title}'. {backstory}{fact_lead} "
                f"Those who encounter me in our story recognize me as {traits_str}."
            )
            char_sentiment = sentiment_service.analyze(reply)
            return {
                "engine": "classical_nlp",
                "character": char_name,
                "response": reply,
                "user_sentiment": user_sentiment,
                "character_sentiment": char_sentiment,
                "citations": [],
                "engine_details": {
                    "method": "Structured Slot-Filling Dossier Synthesis",
                    "intent": "identity_inquiry",
                    "similarity_score": 1.0,
                    "model_used": "None (Pure Classical NLP)",
                    "explanation": "Classified identity inquiry. Formulated direct first-person introduction by slot-filling learned character attributes (role, backstory, traits)."
                }
            }

        # Handle Intent 3: Well-being & Emotional Check
        if intent == "wellbeing":
            compound = user_sentiment.get("compound", 0.0)
            if compound < -0.2:
                reply = (
                    f"I sense heaviness and concern in your words, my friend. As {char_name}, "
                    f"I know well the trials and uncertainties that life presents. "
                    f"Speak openly of what troubles you, and let us consider it together."
                )
            else:
                reply = (
                    f"I stand firm in my spirit and role within '{book_title}'. "
                    f"As {char_name}, I am here beside you as you read our tale. "
                    f"How fares your own journey and thoughts today?"
                )
            char_sentiment = sentiment_service.analyze(reply)
            return {
                "engine": "classical_nlp",
                "character": char_name,
                "response": reply,
                "user_sentiment": user_sentiment,
                "character_sentiment": char_sentiment,
                "citations": [],
                "engine_details": {
                    "method": "NLTK VADER Affective Rule-Based Synthesis",
                    "intent": "wellbeing_sentiment",
                    "similarity_score": round(abs(compound), 4),
                    "model_used": "None (Pure Classical NLP)",
                    "explanation": f"Evaluated query with NLTK VADER sentiment analysis (compound score: {compound}). Selected empathetic affective template matching character's persona."
                }
            }

        # Handle Intent 4: Character Relationships
        if intent == "relationship" and target_entity:
            rel_name = target_entity.get("name", "them")
            rel_desc = char_info.get("relationships", "")
            reply = (
                f"Regarding {rel_name}, they hold a meaningful place in my story. "
                f"{rel_desc} Our interactions and shared moments shape much of what unfolds in '{book_title}'."
            )
            char_sentiment = sentiment_service.analyze(reply)
            return {
                "engine": "classical_nlp",
                "character": char_name,
                "response": reply,
                "user_sentiment": user_sentiment,
                "character_sentiment": char_sentiment,
                "citations": [],
                "engine_details": {
                    "method": "Entity Co-Occurrence & Relationship Graph Lookup",
                    "intent": "character_relationship",
                    "similarity_score": 0.95,
                    "model_used": "None (Pure Classical NLP)",
                    "explanation": f"Recognized entity '{rel_name}' from the book's character registry. Synthesized relationship context in character's first-person voice."
                }
            }

        # Handle Intent 5: Topical / Thematic Book Retrieval
        kw = self.clean_query_keywords(query, char_name)
        retrieved_passages = rag_service.retrieve(query, character_name=char_name, top_k=3)

        top_hit = retrieved_passages[0] if retrieved_passages else None
        top_score = top_hit.get("relevance_score", 0.0) if top_hit else 0.0

        if top_hit and top_score >= 0.05 and kw:
            matched_text = top_hit.get("text", "")
            salient_sentence = self.extract_salient_sentence_or_quote(matched_text, kw)
            clean_quote = salient_sentence.strip(" '\"“”")

            reply = (
                f"Regarding your inquiry, my story and words in '{book_title}' address this directly:\n\n"
                f"\"{clean_quote}\"\n\n"
                f"Reflect upon this wisdom as it relates to your own path."
            )
            char_sentiment = sentiment_service.analyze(reply)
            return {
                "engine": "classical_nlp",
                "character": char_name,
                "response": reply,
                "user_sentiment": user_sentiment,
                "character_sentiment": char_sentiment,
                "citations": [
                    {
                        "chunk_id": p.get("chunk_id"),
                        "relevance_score": p.get("relevance_score"),
                        "snippet": p.get("text", "")[:250] + "..."
                    }
                    for p in retrieved_passages
                ],
                "engine_details": {
                    "method": "TF-IDF Vector Space Model + Salient Dialogue Extraction",
                    "intent": "topical_information_retrieval",
                    "similarity_score": top_score,
                    "matched_quote": clean_quote,
                    "chunk_id": top_hit.get("chunk_id"),
                    "model_used": "None (Pure Classical NLP)",
                    "explanation": f"Vectorized query into TF-IDF space and computed cosine similarity across book chunks. Matched Passage #{top_hit.get('chunk_id')} with cosine score {top_score:.4f}. Extracted the most salient sentence without neural token generation."
                }
            }
        else:
            # Out of Domain Rejection
            topics_fallback = traits_str or "the journey and struggles of our story"
            reply = (
                f"The pages of '{book_title}' do not record my speaking on '{query}'. "
                f"My experiences and counsel in our story center upon {topics_fallback}. "
                f"Ask me about these matters, and I shall share my perspective with you."
            )
            char_sentiment = sentiment_service.analyze(reply)
            return {
                "engine": "classical_nlp",
                "character": char_name,
                "response": reply,
                "user_sentiment": user_sentiment,
                "character_sentiment": char_sentiment,
                "citations": [],
                "engine_details": {
                    "method": "TF-IDF Cosine Similarity Threshold Filtering",
                    "intent": "out_of_domain",
                    "similarity_score": top_score,
                    "model_used": "None (Pure Classical NLP)",
                    "explanation": f"TF-IDF cosine similarity ({top_score:.4f}) fell below the semantic threshold for book contents. Classical NLP domain guard safely declined out-of-universe question."
                }
            }

