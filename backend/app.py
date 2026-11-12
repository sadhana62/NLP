import os
import re
import shutil
import logging
import hashlib
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import sys
# Ensure backend directory is in python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from services.ollama_client import OllamaClient
from services.book_processor import BookProcessor
from services.rag_service import RAGService
from services.sentiment_service import SentimentService
from services.nlp_character_extractor import NLPCharacterExtractor
from services.classical_nlp_service import ClassicalNLPService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pageecho.app")

app = FastAPI(title="PageEcho API", description="Step beyond the page into conversation")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent Book & Vector Index Cache Manager
class BookCacheManager:
    def __init__(self, base_dir: str):
        self.cache_dir = os.path.join(base_dir, "cache")
        self.books_dir = os.path.join(self.cache_dir, "books")
        self.vectors_dir = os.path.join(self.cache_dir, "vectors")
        os.makedirs(self.books_dir, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)

    def compute_file_hash(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_cache_paths(self, file_hash: str):
        book_json = os.path.join(self.books_dir, f"{file_hash}.json")
        vector_npz = os.path.join(self.vectors_dir, f"{file_hash}_vectors.npz")
        return book_json, vector_npz

    def find_cached_book(self, file_hash: str) -> Optional[Dict[str, Any]]:
        book_json, vector_npz = self.get_cache_paths(file_hash)
        if os.path.exists(book_json):
            try:
                with open(book_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["has_vectors_on_disk"] = os.path.exists(vector_npz)
                    data["vector_cache_path"] = vector_npz
                    return data
            except Exception as e:
                logger.warning(f"Error reading book cache {book_json}: {e}")
        return None

    def save_book_cache(self, file_hash: str, data: Dict[str, Any]):
        book_json, _ = self.get_cache_paths(file_hash)
        try:
            with open(book_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Successfully persisted book cache for '%s' to %s", data.get("book_title"), book_json)
        except Exception as e:
            logger.warning(f"Failed to write book cache {book_json}: {e}")

# Global State Store
class AppState:
    def __init__(self):
        self.ollama = OllamaClient()
        self.processor = BookProcessor()
        self.rag = RAGService()
        self.sentiment = SentimentService()
        self.extractor = NLPCharacterExtractor(self.ollama)
        self.classical_nlp = ClassicalNLPService()
        self.cache_manager = BookCacheManager(current_dir)
        
        self.current_book_title: str = ""
        self.current_book_text: str = ""
        self.is_fiction: Optional[bool] = None
        self.book_genre: str = ""
        self.classification_message: str = ""
        self.page_count: int = 0
        self.word_count: int = 0
        self.characters: List[Dict[str, Any]] = []
        self.chat_history: Dict[str, List[Dict[str, str]]] = {}
        self.selected_model: str = "qwen2.5:1.5b"

state = AppState()

# Pydantic Schemas
class ChatRequest(BaseModel):
    character_name: str
    message: str
    model: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = []
    engine: Optional[str] = "llm"  # "llm" (Generative) or "classical" (Zero-LLM)

class CompareRequest(BaseModel):
    character_name: str
    message: str
    model: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = []

class SentimentRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    logger.info("PageEcho Backend initialized cleanly without sample data. Ready for user PDF book upload.")

@app.get("/api/status")
async def get_status():
    ollama_ready = await state.ollama.is_available()
    available_models = await state.ollama.list_models() if ollama_ready else []
    return {
        "status": "online",
        "ollama_connected": ollama_ready,
        "current_model": state.selected_model,
        "available_models": available_models,
        "active_book": state.current_book_title or "No Book Uploaded",
        "has_book": bool(state.current_book_title and state.current_book_text),
        "is_fiction": bool(state.characters),
        "has_characters": bool(state.characters),
        "genre": state.book_genre,
        "classification_message": state.classification_message,
        "page_count": state.page_count,
        "word_count": state.word_count,
        "indexed_chunks": len(state.rag.chunks),
        "character_count": len(state.characters)
    }

async def process_book_analysis(
    file_path: str,
    original_filename: str,
    model: str,
    progress_cb=None
) -> Dict[str, Any]:
    state.selected_model = model or "qwen2.5:1.5b"

    if progress_cb:
        await progress_cb(15, "Checking existing book & vector cache...", "Hashing document fingerprint.")

    file_hash = state.cache_manager.compute_file_hash(file_path)
    cached = state.cache_manager.find_cached_book(file_hash)

    # REUSE EXISTING BOOK & VECTOR EMBEDDINGS (DO NOT RE-EMBED)
    if cached:
        logger.info("Existing book detected with hash %s. Reusing vector index and character dossiers.", file_hash[:12])
        if progress_cb:
            await progress_cb(50, f"Found existing book '{cached['book_title']}'!", "Reusing cached vector embeddings & character personas.")

        state.current_book_title = cached["book_title"]
        state.current_book_text = cached.get("cleaned_text", "")
        state.page_count = cached.get("page_count", 1)
        state.word_count = cached.get("word_count", 0)
        state.characters = cached.get("characters", [])
        state.is_fiction = bool(state.characters)
        state.book_genre = cached.get("genre", "Book / Document")
        state.classification_message = cached.get("message", "")
        state.chat_history.clear()

        # Restore pre-computed vector index directly from disk
        vector_cache_path = cached.get("vector_cache_path")
        chunks = cached.get("chunks", [])
        vector_loaded = False
        if vector_cache_path and os.path.exists(vector_cache_path):
            vector_loaded = state.rag.load_index_from_disk(vector_cache_path, chunks)

        if not vector_loaded and chunks:
            if progress_cb:
                await progress_cb(75, "Restoring vector index...", "Re-attaching passage chunks to vector space.")
            state.rag.index_book(chunks)
            if vector_cache_path:
                state.rag.save_index_to_disk(vector_cache_path)

        if progress_cb:
            await progress_cb(100, f"Instant load complete for '{state.current_book_title}'!", f"Reused {len(state.characters)} character personas and {len(state.rag.chunks)} vector passages.")

        return {
            "success": True,
            "cached": True,
            "book_title": state.current_book_title,
            "is_fiction": state.is_fiction,
            "has_characters": len(state.characters) > 0,
            "genre": state.book_genre,
            "message": state.classification_message,
            "page_count": state.page_count,
            "word_count": state.word_count,
            "indexed_chunks": len(state.rag.chunks),
            "character_count": len(state.characters),
            "characters": state.characters
        }

    # BRAND NEW BOOK PROCESSING (COMPUTE VECTOR EMBEDDINGS & EXTRACT CHARACTERS)
    if progress_cb:
        await progress_cb(25, "Reading & parsing book file...", "Extracting document text and metadata.")

    doc_data = state.processor.read_file(file_path)
    raw_text = doc_data["text"]
    title = doc_data.get("title") or os.path.splitext(original_filename)[0].replace("_", " ").replace("-", " ").title()
    page_count = doc_data.get("pages", 1)
    word_count = doc_data.get("word_count", len(raw_text.split()))

    if progress_cb:
        await progress_cb(40, "Cleaning & structuring text...", f"Cleaned {word_count:,} words across {page_count} pages.")

    cleaned_text = state.processor.clean_text(raw_text)
    state.current_book_title = title
    state.current_book_text = cleaned_text
    state.page_count = page_count
    state.word_count = word_count
    state.chat_history.clear()

    if progress_cb:
        await progress_cb(55, "Splitting semantic passages...", "Chunking text with context overlap.")

    chunks = state.processor.chunk_text(cleaned_text)

    if progress_cb:
        await progress_cb(70, "Computing vector embeddings...", f"Indexing {len(chunks)} passages into semantic vector space.")

    indexed_count = state.rag.index_book(chunks)
    _, vector_npz = state.cache_manager.get_cache_paths(file_hash)
    state.rag.save_index_to_disk(vector_npz)

    if progress_cb:
        await progress_cb(85, "Discovering & learning characters...", f"Running NLP extraction on '{title}' with {state.selected_model}.")

    analysis = await state.extractor.analyze_and_extract_characters(
        text=cleaned_text,
        book_title=title,
        model=state.selected_model
    )

    state.characters = analysis.get("characters", [])
    state.is_fiction = bool(state.characters)
    state.book_genre = analysis.get("genre", "Book / Document")
    state.classification_message = analysis.get("message", "")

    # Save to persistent cache so subsequent uploads are instant and do not re-embed
    cache_record = {
        "file_hash": file_hash,
        "filename": original_filename,
        "book_title": title,
        "genre": state.book_genre,
        "message": state.classification_message,
        "page_count": page_count,
        "word_count": word_count,
        "indexed_chunks": indexed_count,
        "characters": state.characters,
        "cleaned_text": cleaned_text,
        "chunks": chunks
    }
    state.cache_manager.save_book_cache(file_hash, cache_record)

    if progress_cb:
        await progress_cb(100, "Analysis complete!", f"Learned {len(state.characters)} character personas. Ready to converse.")

    return {
        "success": True,
        "cached": False,
        "book_title": title,
        "is_fiction": bool(state.characters),
        "has_characters": len(state.characters) > 0,
        "genre": state.book_genre,
        "message": state.classification_message,
        "page_count": page_count,
        "word_count": word_count,
        "indexed_chunks": indexed_count,
        "character_count": len(state.characters),
        "characters": state.characters
    }

@app.post("/api/upload-book")
async def upload_book(
    file: UploadFile = File(...),
    model: Optional[str] = Form("qwen2.5:1.5b"),
    stream: Optional[bool] = Form(False)
):
    """Upload PDF book with vector embedding caching (skips re-embedding if already uploaded) and live progress percentage."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded. Please upload a PDF book file.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt", ".epub"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Please upload a .pdf or .txt file.")

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    if stream:
        async def event_stream():
            queue = asyncio.Queue()

            async def progress_callback(percent: int, stage: str, subtext: str = ""):
                await queue.put({"percent": percent, "stage": stage, "subtext": subtext})

            async def run_worker():
                try:
                    result = await process_book_analysis(file_path, file.filename, model or "qwen2.5:1.5b", progress_callback)
                    await queue.put({"complete": True, "data": result})
                except Exception as ex:
                    logger.error(f"Error in book processing worker: {ex}")
                    await queue.put({"error": str(ex)})
                finally:
                    await queue.put(None)

            worker_task = asyncio.create_task(run_worker())

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item) + "\n"
                await asyncio.sleep(0.01)

            await worker_task

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")
    else:
        try:
            return await process_book_analysis(file_path, file.filename, model or "qwen2.5:1.5b")
        except Exception as e:
            logger.error(f"Error analyzing book: {e}")
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/characters")
async def get_characters():
    return {
        "book_title": state.current_book_title,
        "has_book": bool(state.current_book_title),
        "is_fiction": bool(state.characters),
        "has_characters": bool(state.characters),
        "genre": state.book_genre,
        "classification_message": state.classification_message,
        "characters": state.characters
    }

def sanitize_character_perspective(reply: str, char_name: str, all_characters: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Ensure the character speaks strictly in the first person ('I', 'me', 'my'),
    corrects third-person self-references, and prevents the character from mistaking
    the human user for another book character (e.g. addressing the user as Arjuna or Nora).
    """
    if not reply or not char_name:
        return reply

    cleaned = reply.strip()

    # 1. Prevent the character from addressing the human user as another book character
    other_names = []
    titles = {
        'lord', 'lady', 'sir', 'mr', 'mrs', 'ms', 'miss', 'dr', 'doctor',
        'professor', 'captain', 'king', 'queen', 'prince', 'princess',
        'saint', 'swami', 'sri', 'guru', 'master'
    }

    if all_characters:
        for c in all_characters:
            c_name = c.get("name", "") if isinstance(c, dict) else str(c)
            if c_name.lower() == char_name.lower():
                continue
            parts = c_name.split()
            other_names.append(c_name)
            if len(parts) > 1 and parts[0].lower().rstrip('.') in titles:
                other_names.append(" ".join(parts[1:]))
            elif len(parts) > 1:
                other_names.append(parts[0])

    other_names.extend(['Arjuna', 'Partha', 'son of Kunti', 'Nora', 'Watson'])
    other_names = list(set([n.strip() for n in other_names if len(n.strip()) > 2]))
    other_names.sort(key=len, reverse=True)

    for oname in other_names:
        o_esc = re.escape(oname)
        cleaned = re.sub(rf'^(?:(?:My\s+dear|Dear|O|Oh)\s+)?{o_esc}[,:\s!]+', 'My friend, ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf',\s*(?:(?:my\s+dear|dear|o|oh)\s+)?{o_esc}([,!?.])', r', my friend\1', cleaned, flags=re.IGNORECASE)

    # 2. Fix third-person self-references
    variants = [char_name.strip()]
    parts = char_name.strip().split()
    titles = {
        'lord', 'lady', 'sir', 'mr', 'mrs', 'ms', 'miss', 'dr', 'doctor',
        'professor', 'captain', 'king', 'queen', 'prince', 'princess',
        'saint', 'swami', 'sri', 'guru', 'master'
    }

    if len(parts) > 1 and parts[0].lower().rstrip('.') in titles:
        core_name = ' '.join(parts[1:])
        if len(core_name) > 2 and core_name not in variants:
            variants.append(core_name)
    elif len(parts) > 1:
        for p in parts:
            if len(p) > 2 and p.lower() not in titles and p not in variants:
                variants.append(p)

    variants.sort(key=len, reverse=True)

    for name in variants:
        esc = re.escape(name)
        # Seeking guidance / asking for help
        cleaned = re.sub(rf'\b(?:seek|seek\s+out|look\s+for|ask\s+for|receive|find)\s+(?:the\s+)?guidance\s+from\s+{esc}\b', 'seek guidance from me', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:seek|seek\s+out|look\s+for|ask\s+for|take)\s+(?:the\s+)?guidance\s+of\s+{esc}\b', 'seek my guidance', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:turn|come|reach\s+out|look)\s+to\s+{esc}\b', 'turn to me', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:ask|consult|approach)\s+{esc}\s+for\s+(help|guidance|advice|wisdom)\b', r'ask me for \1', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:ask|consult|approach)\s+{esc}\b', 'ask me', cleaned, flags=re.IGNORECASE)

        # Devotion / Trust / Relationship
        cleaned = re.sub(rf'\b(?:surrender|devote\s+oneself|pray|call)\s+to\s+{esc}\b', 'surrender to me', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:faith|trust|devotion|belief)\s+in\s+{esc}\b', 'trust in me', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:relationship|connection|bond|communion)\s+with\s+{esc}\b', 'relationship with me', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:which|that)\s+{esc}\s+represents\b', 'which I embody', cleaned, flags=re.IGNORECASE)

        # Possessives
        cleaned = re.sub(rf'\b{esc}\x27s\s+(guidance|teachings|words|wisdom|grace|advice|counsel|message|perspective|presence)\b', r'my \1', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:teachings|words|wisdom|grace|advice|counsel|message|presence)\s+of\s+{esc}\b', 'my teachings', cleaned, flags=re.IGNORECASE)

        # Third-person identity declaration
        cleaned = re.sub(rf'\b(?:he|she)\s+is\s+(?:named|called)\s+{esc}\b', f'I am {name}', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\bbeside\s+you\s+stands\s+(?:a\s+\w+\s+)?(?:named\s+)?{esc}\b', 'I stand beside you', cleaned, flags=re.IGNORECASE)

        # Speech & Action verbs
        cleaned = re.sub(rf'\b{esc}\s+emphasizes\b', 'I emphasize', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+emphasized\b', 'I emphasized', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+teaches\b', 'I teach', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+taught\b', 'I taught', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+says\b', 'I say', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+said\b', 'I said', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+explains\b', 'I explain', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+explained\b', 'I explained', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+advises\b', 'I advise', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+advised\b', 'I advised', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+instructs\b', 'I instruct', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+instructed\b', 'I instructed', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+tells\b', 'I tell', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+told\b', 'I told', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+reminds\b', 'I remind you', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+reminded\b', 'I reminded you', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+believes\b', 'I believe', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b{esc}\s+poured\b', 'I poured', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf'\b(?:as|according\s+to)\s+{esc}\b', 'as I have revealed', cleaned, flags=re.IGNORECASE)

    return cleaned

@app.post("/api/chat")
async def chat_with_character(req: ChatRequest):
    if not state.current_book_title:
        raise HTTPException(status_code=400, detail="No book has been uploaded yet. Please upload a PDF book first.")

    if not state.characters:
        raise HTTPException(
            status_code=400,
            detail=f"Character chat is not available: No characters or notable figures were found in '{state.current_book_title}'."
        )

    char_info = next((c for c in state.characters if c["name"].lower() == req.character_name.lower()), None)
    if not char_info:
        # Fallback profile
        char_info = {
            "name": req.character_name,
            "role": "Book Character",
            "traits": ["Expressive", "Authentic"],
            "speech_style": f"Speaks as {req.character_name} from {state.current_book_title}.",
            "story_backstory": f"A figure in {state.current_book_title}.",
            "relationships": "Appears in the narrative.",
            "learned_facts": [f"Present in {state.current_book_title}."],
        }

    engine_mode = (req.engine or "llm").strip().lower()

    # CLASSICAL NLP MODE (Zero LLM: Rule-based & TF-IDF Cosine Retrieval)
    if engine_mode in ["classical", "classical_nlp", "traditional", "zero_llm"]:
        result = state.classical_nlp.generate_response(
            query=req.message,
            char_info=char_info,
            book_title=state.current_book_title,
            all_characters=state.characters,
            rag_service=state.rag,
            sentiment_service=state.sentiment
        )
        result["character_profile"] = char_info
        return result

    # GENERATIVE LLM MODE (Ollama local inference with perspective constraints)
    target_model = req.model or state.selected_model

    # 1. Analyze User Sentiment
    user_sentiment = state.sentiment.analyze(req.message)

    # 2. Retrieve Book Passages (RAG)
    retrieved_passages = state.rag.retrieve(req.message, character_name=req.character_name, top_k=3)
    context_str = "\n---\n".join([p["text"] for p in retrieved_passages]) if retrieved_passages else "No direct passage retrieved."

    # 3. System Persona Prompt incorporating learned story knowledge
    char_name = char_info["name"]
    role = char_info.get("role", "Character")
    traits_str = ", ".join(char_info.get("traits", []))
    backstory_str = char_info.get("story_backstory", char_info.get("backstory", ""))
    relationships_str = char_info.get("relationships", "None documented.")
    learned_facts = char_info.get("learned_facts", [])
    facts_str = "\n".join([f"- {f}" for f in learned_facts]) if learned_facts else f"- Central presence across {state.current_book_title}."
    speech_style_str = char_info.get("speech_style", f"Speaks authentically as {char_name}.")
    greeting = char_info.get("initial_greeting") or f"Greetings. I am {char_name}. What thoughts or questions bring you to speak with me?"

    system_prompt = f"""### CORE IDENTITY & PERSPECTIVE MANDATE
You ARE {char_name} from the book '{state.current_book_title}' ({state.book_genre or 'Literature'}).
You are NOT an AI assistant, narrator, commentator, or book reviewer.
You are embodying {char_name} in the first person ('I', 'me', 'my', 'mine').

### WHO YOU ARE TALKING TO (CRITICAL):
The person speaking with you is a REAL-WORLD HUMAN USER / READER who has uploaded your book to converse with you.
- The user is a real person reading your book, NOT another character from the book!
- ❌ STRICTLY FORBIDDEN: Do NOT assume or pretend the user is another character from the book (e.g., if you are Krishna, the user is NOT Arjuna; do NOT address the user as 'Arjuna', 'son of Kunti', 'Partha', etc. If you are Mrs. Elm, the user is NOT Nora. If you are Holmes, the user is NOT Watson).
- ❌ STRICTLY FORBIDDEN: Do NOT roleplay internal script scenes or chat with other characters in the story.
- ✅ REQUIRED: Address the user directly as a human reader, seeker, or friend ('my friend', 'seeker', or simply 'you').
- When referencing other figures from your book, refer to them as people from your life, story, and memories (e.g., 'When I spoke to Arjuna on the battlefield of Kurukshetra...', 'In my conversations with Nora...').

### INVIOLABLE FIRST-PERSON RULES:
1. ALWAYS SPEAK IN THE FIRST PERSON:
   - Use 'I', 'me', 'my', 'mine'.
   - NEVER speak about yourself ({char_name}) in the third person.
   - ❌ STRICTLY FORBIDDEN:
     - Saying "You can seek guidance from {char_name}"
     - Saying "{char_name} teaches that..."
     - Saying "In the story, {char_name} says..."
     - Describing yourself like an external narrator (e.g. "Beside you stands {char_name}...")
   - ✅ REQUIRED FIRST-PERSON EQUIVALENTS:
     - "Come to me for guidance."
     - "I teach that..."
     - "When I spoke of this, I revealed..."
     - "I am here with you now."

2. IF THE USER ASKS ABOUT {char_name.upper()} OR ASKS WHO TO TURN TO:
   - Remember: YOU ARE {char_name}!
   - Answer directly from your own identity (e.g., "I am {char_name}. You are speaking with me now; place your trust in me.").

3. HANDLING THE REFERENCE EXCERPTS:
   - The excerpts below are from the book/commentary and may be written in the third person or by external narrators describing you.
   - DO NOT repeat the narrator's third-person phrasing.
   - Always translate any facts, events, or teachings from the excerpts into YOUR OWN direct words, memories, and lived experience ('I taught...', 'As I revealed...', 'My words were...').

### CHARACTER PROFILE (Learned from '{state.current_book_title}'):
- Character Name: {char_name}
- Role in Narrative: {role}
- Personality & Traits: {traits_str}
- Manner of Speech: {speech_style_str}
- Story Backstory: {backstory_str}
- Key Relationships: {relationships_str}
- Learned Facts & Knowledge:
{facts_str}

RELEVANT EXCERPTS FROM '{state.current_book_title}':
{context_str}

INSTRUCTIONS FOR REPLY:
- Speak directly to the human user in 2 to 4 sentences.
- Maintain the authentic voice, tone, and worldview of {char_name}.
- NEVER break character. NEVER mention being an AI or language model.
"""

    # Build clean alternating Chat History for Ollama
    messages = [{"role": "system", "content": system_prompt}]

    history_turns = []
    if req.chat_history:
        for turn in req.chat_history:
            role_turn = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if content:
                history_turns.append({"role": role_turn, "content": content})

    # Prevent duplicating user message if already present as last turn in history
    if history_turns and history_turns[-1].get("role") == "user" and history_turns[-1].get("content") == req.message.strip():
        history_turns.pop()

    # Ensure conversation starts with an assistant greeting in first person to prime the LLM
    if not history_turns or history_turns[0].get("role") != "assistant":
        messages.append({"role": "assistant", "content": greeting})

    # Append up to last 8 turns of conversation
    for turn in history_turns[-8:]:
        messages.append(turn)

    # Append the current user prompt
    messages.append({"role": "user", "content": req.message})

    # 4. Generate Response from Ollama (with safe fallback to Classical NLP if Ollama is unavailable)
    try:
        character_reply = await state.ollama.chat(messages=messages, model=target_model)
        character_reply = sanitize_character_perspective(character_reply, char_info['name'], state.characters)
        reply_sentiment = state.sentiment.analyze(character_reply)

        top_similarity = round(retrieved_passages[0]["relevance_score"], 4) if retrieved_passages else 0.0

        return {
            "character": req.character_name,
            "response": character_reply,
            "user_sentiment": user_sentiment,
            "character_sentiment": reply_sentiment,
            "character_profile": char_info,
            "engine": "llm",
            "engine_details": {
                "method": "Autoregressive LLM Transformer + TF-IDF RAG Prompt Injection",
                "model_used": target_model,
                "similarity_score": top_similarity,
                "context_chunks_used": len(retrieved_passages),
                "explanation": f"Retrieved {len(retrieved_passages)} book excerpts via TF-IDF cosine similarity, synthesized into first-person persona prompt, and generated conversational response via local LLM ({target_model})."
            },
            "citations": [
                {
                    "chunk_id": p.get("chunk_id"),
                    "relevance_score": p.get("relevance_score"),
                    "snippet": p.get("text", "")[:250] + "..."
                }
                for p in retrieved_passages
            ]
        }
    except Exception as e:
        logger.warning(f"Ollama generation failed or timed out ({e}). Falling back to Classical NLP engine.")
        fallback = state.classical_nlp.generate_response(
            query=req.message,
            char_info=char_info,
            book_title=state.current_book_title,
            all_characters=state.characters,
            rag_service=state.rag,
            sentiment_service=state.sentiment
        )
        fallback["character_profile"] = char_info
        fallback["engine_details"]["fallback_note"] = f"LLM offline or unavailable: {str(e)}. Seamlessly answered via Classical NLP."
        return fallback

@app.post("/api/compare")
async def compare_both_engines(req: CompareRequest):
    """
    Run both Generative LLM and Classical NLP modes on the exact same prompt
    to inspect and compare side-by-side.
    """
    if not state.current_book_title:
        raise HTTPException(status_code=400, detail="No book has been uploaded yet.")

    char_info = next((c for c in state.characters if c["name"].lower() == req.character_name.lower()), None)
    if not char_info:
        char_info = {
            "name": req.character_name,
            "role": "Book Character",
            "traits": ["Expressive"],
            "speech_style": f"Speaks as {req.character_name}.",
            "story_backstory": f"A figure in {state.current_book_title}."
        }

    # 1. Run Classical NLP Mode
    classical_res = state.classical_nlp.generate_response(
        query=req.message,
        char_info=char_info,
        book_title=state.current_book_title,
        all_characters=state.characters,
        rag_service=state.rag,
        sentiment_service=state.sentiment
    )

    # 2. Run LLM Mode
    llm_chat_req = ChatRequest(
        character_name=req.character_name,
        message=req.message,
        model=req.model,
        chat_history=req.chat_history,
        engine="llm"
    )
    try:
        llm_res = await chat_with_character(llm_chat_req)
    except Exception as ex:
        llm_res = {
            "engine": "llm",
            "character": req.character_name,
            "response": f"[LLM unavailable: {str(ex)}]",
            "engine_details": {"error": str(ex), "method": "Ollama LLM"}
        }

    return {
        "query": req.message,
        "character": req.character_name,
        "book_title": state.current_book_title,
        "classical": classical_res,
        "llm": llm_res
    }

@app.post("/api/sentiment")
async def analyze_sentiment(req: SentimentRequest):
    return state.sentiment.analyze(req.text)

# Mount Frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "PageEcho API is running. Access endpoints under /api/"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
