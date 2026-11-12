# PageEcho — Step Beyond the Page into Conversation

PageEcho is an intelligent literary exploration system that lets you step beyond the page into direct conversation with characters from any book. It automatically discovers and extracts primary characters using linguistic heuristics, learns their psychological dossiers, and enables authentic 1-on-1 first-person conversations with any chosen character.

---

## Key Features

- **Dual Mode Switch (LLM vs. Classical NLP):** Toggle between **Generative LLM Mode** (local Ollama neural token generation with RAG) and **Classical NLP Mode** (pure Information Retrieval, TF-IDF cosine similarity, regex intent classification, and verbatim quote extraction with zero neural networks or LLMs).
- **Side-by-Side Architecture Inspector:** Interactive comparison test bench (`⚖️ Compare Engines`) allows testing any prompt across both paradigms simultaneously with real-time diagnostic breakdowns.
- **Automated NLP Character Extraction:** Heuristic and linguistic entity recognition detects character candidates, dialogue attributions, mid-sentence proper nouns, and standardized honorifics (`Lord`, `Dr.`, `King`, `Sri`, `Swami`, etc.).
- **Authentic 1-on-1 First-Person Roleplay:** Selected characters speak strictly in the first person (`I`, `me`, `my`), address the user as a real-world human reader/seeker, and never confuse the user with in-book characters (e.g., Lord Krishna will not address the user as Arjuna).
- **Persistent Vector & Dossier Caching:** When a book is processed once, its vector index and character dossiers are saved to disk (`backend/cache/`). Re-uploading the same book skips re-indexing and Ollama calls entirely, loading in under **0.05 seconds**.
- **Live Progress Percentage:** Real-time event streaming displays an animated progress bar (15% $\to$ 40% $\to$ 70% $\to$ 100%) showing the exact stage of document analysis.
- **Embedded Semantic RAG:** Passage chunking (800 characters with 150-character contextual overlap) indexed into an L2-normalized TF-IDF vector space with smooth IDF for rapid cosine similarity retrieval.
- **VADER Sentiment Analysis:** Real-time polarity scoring (positive, neutral, negative, compound) for both user messages and character responses.
- **Voice Synthesis:** Integrated Web Speech API for in-browser text-to-speech character voicing.

---

## System Architecture

```
[Uploaded Book (PDF/TXT)]
           │
           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 1. Ingestion & SHA-256 Fingerprinting                   │
 │    Hashes file content and checks persistent disk cache │
 └─────────────────────────┬───────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
        [Cache Hit]                 [Cache Miss]
             │                           │
             ▼                           ▼
 ┌──────────────────────┐   ┌────────────────────────────────────────┐
 │ Instant Load (~40ms) │   │ 2. Parsing & Text Normalization        │
 │ Reuses .npz vectors  │   │    PyPDF text extraction, cleans line  │
 │ Reuses dossiers      │   │    breaks, ligatures & broken hyphens  │
 └───────────┬──────────┘   └───────────────────┬────────────────────┘
             │                                  │
             │                                  ▼
             │              ┌────────────────────────────────────────┐
             │              │ 3. Chunking & Vector Indexing          │
             │              │    Sliding window (800 chars / 150 ovl)│
             │              │    Computes TF-IDF matrix + Smooth IDF │
             │              │    Persists vectors to cache/{hash}.npz│
             │              └───────────────────┬────────────────────┘
             │                                  │
             │                                  ▼
             │              ┌────────────────────────────────────────┐
             │              │ 4. NLP Heuristics & Persona Learning   │
             │              │    Dialogue attribution & Title regex  │
             │              │    Synthesizes dossiers & voice rules  │
             │              │    Persists book to cache/{hash}.json  │
             │              └───────────────────┬────────────────────┘
             │                                  │
             └──────────────────┬───────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 5. Conversational RAG & Persona Engine (Chat Turn)      │
 │    - Query expansion: "{character_name} {user_message}" │
 │    - Cosine similarity passage retrieval                │
 │    - Identity Mandate: First-person translation rules   │
 │    - Reader Context: User != in-book characters         │
 │    - Local LLM generation turn                          │
 │    - Post-generation Perspective Sanitizer Guardrail    │
 └─────────────────────────────────────────────────────────┘
```

---

## Technical Pipeline Details

### 1. Book Ingestion & Normalization (`backend/services/book_processor.py`)
- Extracts plain text from `.pdf`, `.txt`, and `.epub` formats.
- Corrects line-broken hyphenated words (`someth-\ning` $\to$ `something`).
- Normalizes unicode quotation marks and dashes.
- Segments text into semantic passages with 150-character overlaps to preserve context across chunk boundaries.

### 2. Persistent Caching (`backend/cache/`)
- Computes SHA-256 checksum of uploaded files.
- Caches learned character dossiers to `backend/cache/books/{hash}.json`.
- Caches pre-computed TF-IDF matrices, IDF weights, and vocabulary dictionaries to `backend/cache/vectors/{hash}_vectors.npz` using `np.savez_compressed()`.
- Re-uploading an existing book bypasses text extraction, chunking, vector computation, and Ollama extraction completely.

### 3. NLP Character Extraction (`backend/services/nlp_character_extractor.py`)
- **Linguistic Heuristics:**
  - Scans for title-name patterns (`Lord`, `King`, `Dr.`, `Sri`, `Swami`, etc.).
  - Extracts dialogue speaker attributions (`[Name] said`, `replied [Name]`).
  - Matches proper nouns in mid-sentence positions to eliminate common capitalized stopwords.
  - Consolidates single names into established full names.
- **Persona Dossier Synthesis:**
  - Gathers contextual snippets where the character appears.
  - Extracts narrative role, psychological traits, speech cadence, backstory, and an authentic first-person greeting welcoming the human reader.

### 4. Vector Embedding & RAG Search (`backend/services/rag_service.py`)
- Employs an in-memory TF-IDF Vector Space Model:
  $$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{total terms in } d}$$
  $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1.0$$
- L2 normalization ensures rapid cosine similarity computation via vector dot-products.
- Retrieves the top 3 relevant passages for each query and provides citations with relevance scores.

### 5. Chat Persona Engine & Perspective Sanitizer (`backend/app.py`)
- **Inviolable First-Person Mandate:** Enforces `'I'`, `'me'`, `'my'`, and bans third-person phrasing (e.g. forbids *"You can seek guidance from [Character]"*).
- **Narrative Perspective Translation:** Directs the model to interpret third-person book commentary from its own lived first-person experience.
- **Reader-Centric Framing:** Informs the character that the conversational partner is a human reader, forbidding the character from addressing the user as an in-book counterpart (e.g., Krishna will not address the user as Arjuna).
- **Perspective Sanitizer Guardrail:** A regex post-processor that intercepts and repairs any third-person slippages or vocative character misaddressing before reaching the frontend.

### 6. Classical NLP Service (`backend/services/classical_nlp_service.py`)
Provides a complete **Zero-LLM Dialogue Engine** using classical NLP techniques:
- **Intent Classification via Heuristics:** Identifies dialogue intent categories without neural classifiers:
  - `greeting`: Welcomes the reader in character's literary voice.
  - `identity_inquiry`: Synthesizes first-person identity from slot-filled character attributes (role, backstory, traits).
  - `wellbeing_sentiment`: Assesses user sentiment polarity with NLTK VADER and responds with empathetic affective templates.
  - `character_relationship`: Links named entities in the query against character co-occurrence graphs.
  - `topical_information_retrieval`: Extracts salient book dialogue and canonical quotes matching user inquiry terms via TF-IDF cosine similarity.
  - `out_of_domain`: Filters out queries whose substantive keywords do not appear in the book, gracefully keeping the character in-universe.
- **Architectural Comparison:**

| Feature | ⚡ Generative LLM Mode | 🏛️ Classical NLP Mode (Zero LLM) |
|---|---|---|
| **Underlying Engine** | Local Ollama Transformer (Qwen 2.5 / Gemma 3) | TF-IDF Vector Space + NLTK VADER + Slot Filling |
| **Hardware Required** | GPU or Multi-Core CPU for Ollama | Lightweight (runs in &lt;5ms on any machine) |
| **Output Style** | Dynamic, conversational, creative roleplay | Verbatim canonical quotes & structured templates |
| **Hallucination Risk** | Low-to-Moderate (mitigated by RAG) | **0% (Zero Hallucination)** — 100% book-grounded |
| **Determinism** | Probabilistic (temperature-dependent) | **100% Deterministic** |
| **Out-of-Domain Handling** | LLM roleplay refusal prompt | Cosine threshold rejection |

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Local [Ollama](https://ollama.com/) instance installed and running

### 1. Install Dependencies
```powershell
cd backend
pip install -r requirements.txt
```

### 2. Pull Required Model in Ollama
```powershell
ollama pull qwen2.5:1.5b
```

### 3. Run the Backend Server
```powershell
cd backend
python app.py
```
The server will start at `http://localhost:8000`.

### 4. Access the Web Application
Open your browser and navigate to:
```
http://localhost:8000
```

---

## Usage Guide

1. **Upload Book:** Click **"Select & Upload PDF Book"** and choose a PDF or TXT file.
2. **Monitor Progress:** Watch the real-time progress bar (15% $\to$ 40% $\to$ 70% $\to$ 100%) as the book is parsed, chunked, vector-indexed, and profiled.
3. **Select a Character:** Browse the detected characters in the sidebar and click a card to open their conversation.
4. **Choose Dialogue Engine Mode:**
   - Click **`⚡ Generative LLM`** in the top navigation bar for fluid conversational turn-taking via Ollama.
   - Click **`🏛️ Classical NLP (Zero LLM)`** to converse using purely classical Information Retrieval, TF-IDF cosine matching, and verbatim quote extraction.
5. **Inspect Side-by-Side Comparison:** Click **`⚖️ Compare Engines`** in the chat header to test any prompt on both engines simultaneously and inspect their diagnostic outputs side-by-side.
6. **Inspect Dossier:** Click **"Learned Dossier"** to review the character's extracted backstory, personality traits, and speech cadence.
7. **Chat:** Ask questions, explore ethical dilemmas, or discuss the book's events. The character will respond in their distinctive literary voice directly to you.
8. **Verify Instant Cache:** Re-upload the same book to see it load instantly in ~0.04 seconds without recalculating embeddings.

---

## Project Structure

```
NLP/
├── backend/
│   ├── app.py                         # FastAPI server, dual-engine endpoints & perspective sanitizer
│   ├── requirements.txt               # Backend dependencies
│   ├── cache/                         # Persistent storage
│   │   ├── books/                     # Cached JSON book dossiers
│   │   └── vectors/                   # Cached .npz TF-IDF vector embeddings
│   ├── services/
│   │   ├── classical_nlp_service.py   # Pure classical NLP dialogue, quote extractor & intent matcher
│   │   ├── book_processor.py          # PDF/TXT extraction, text cleaning & chunking
│   │   ├── nlp_character_extractor.py # Linguistic heuristics & persona extraction
│   │   ├── rag_service.py             # TF-IDF vector index, cosine similarity & disk caching
│   │   ├── sentiment_service.py       # NLTK VADER sentiment scoring
│   │   └── ollama_client.py           # Async HTTP interface to local Ollama
│   └── uploads/                       # Uploaded document storage
├── frontend/
│   ├── index.html                     # UI layout, dual-engine segmented switch & comparison modal
│   ├── style.css                      # Modern typography, progress bar, engine badges & comparison modal
│   └── app.js                         # State management, dual-mode controller & comparative test bench
└── README.md                          # Comprehensive project documentation
```
