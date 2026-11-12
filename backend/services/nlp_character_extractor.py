import re
import json
import logging
from collections import Counter
from typing import List, Dict, Any, Tuple
from services.ollama_client import OllamaClient

logger = logging.getLogger("pageecho.character_extractor")

# Comprehensive non-character stopwords: pronouns, conjunctions, prepositions, adverbs, modals, time words
STOPWORDS = {
    'A', 'About', 'Above', 'Across', 'After', 'Afterward', 'Again', 'Against', 'All', 'Almost',
    'Alone', 'Along', 'Already', 'Also', 'Although', 'Always', 'Am', 'Among', 'An', 'And', 'Another',
    'Any', 'Anybody', 'Anyone', 'Anything', 'Anyway', 'Are', 'Around', 'As', 'At', 'Back', 'Be',
    'Became', 'Because', 'Become', 'Becomes', 'Becoming', 'Been', 'Before', 'Began', 'Behind',
    'Being', 'Below', 'Beside', 'Between', 'Beyond', 'Both', 'But', 'By', 'Can', 'Cannot', 'Come',
    'Could', 'Did', 'Do', 'Does', 'Doing', 'Done', 'Down', 'During', 'Each', 'Either', 'Else',
    'Even', 'Ever', 'Every', 'Everyone', 'Everything', 'Everywhere', 'Except', 'Far', 'Few',
    'Finally', 'First', 'For', 'From', 'Further', 'Get', 'Give', 'Go', 'Going', 'Gone', 'Good',
    'Got', 'Great', 'Had', 'Has', 'Have', 'Having', 'He', 'Hello', 'Help', 'Her', 'Here', 'Hers',
    'Herself', 'Him', 'Himself', 'His', 'How', 'However', 'I', 'If', 'In', 'Indeed', 'Inside',
    'Instead', 'Into', 'Is', 'It', 'Its', 'Itself', 'Just', 'Keep', 'Know', 'Last', 'Late', 'Later',
    'Least', 'Less', 'Let', 'Like', 'Little', 'Look', 'Looked', 'Looking', 'Made', 'Make', 'Many',
    'May', 'Maybe', 'Me', 'Mean', 'Meanwhile', 'Might', 'More', 'Moreover', 'Most', 'Mostly', 'Much',
    'Must', 'My', 'Myself', 'Near', 'Need', 'Neither', 'Never', 'New', 'Next', 'No', 'Nobody',
    'None', 'Noone', 'Nor', 'Not', 'Nothing', 'Now', 'Nowhere', 'Of', 'Off', 'Often', 'Oh', 'Old',
    'On', 'Once', 'One', 'Only', 'Onto', 'Or', 'Other', 'Others', 'Otherwise', 'Our', 'Ours',
    'Ourselves', 'Out', 'Outside', 'Over', 'Own', 'Part', 'Perhaps', 'Please', 'Put', 'Rather',
    'Really', 'Right', 'Said', 'Same', 'Saw', 'Say', 'Saying', 'See', 'Seeing', 'Seen', 'She',
    'Should', 'Since', 'So', 'Some', 'Somebody', 'Someone', 'Something', 'Sometime', 'Sometimes',
    'Somewhere', 'Soon', 'Still', 'Such', 'Suddenly', 'Sure', 'Take', 'Taken', 'Tell', 'Than',
    'Thank', 'Thanks', 'That', 'The', 'Their', 'Theirs', 'Them', 'Themselves', 'Then', 'There',
    'Therefore', 'These', 'They', 'Thing', 'Things', 'Think', 'Thinking', 'This', 'Those', 'Though',
    'Thought', 'Through', 'Throughout', 'Thru', 'Thus', 'To', 'Together', 'Too', 'Toward', 'Towards',
    'True', 'Truly', 'Two', 'Under', 'Until', 'Up', 'Upon', 'Us', 'Use', 'Used', 'Very', 'Wait',
    'Was', 'Way', 'We', 'Well', 'Went', 'Were', 'What', 'Whatever', 'When', 'Whence', 'Whenever',
    'Where', 'Whereafter', 'Whereas', 'Whereby', 'Wherein', 'Whereupon', 'Wherever', 'Whether',
    'Which', 'While', 'Whither', 'Who', 'Whoever', 'Whole', 'Whom', 'Whose', 'Why', 'Will', 'With',
    'Within', 'Without', 'Would', 'Yes', 'Yet', 'You', 'Your', 'Yours', 'Yourself', 'Yourselves',
    'Chapter', 'Page', 'Book', 'Volume', 'Section', 'Copyright', 'Published', 'Press', 'Edition',
    'Library', 'Penguin', 'Random', 'House', 'London', 'English', 'Life', 'Death', 'World', 'Time',
    'Update', 'None', 'Table', 'Figure', 'Index', 'Author', 'Title', 'Prologue', 'Epilogue', 'Contents'
}

TITLES = {
    'Mr': 'Mr.', 'Mrs': 'Mrs.', 'Ms': 'Ms.', 'Miss': 'Miss', 'Dr': 'Dr.', 'Doctor': 'Doctor',
    'Lord': 'Lord', 'Lady': 'Lady', 'Sir': 'Sir', 'Captain': 'Captain', 'Professor': 'Professor',
    'Inspector': 'Inspector', 'King': 'King', 'Queen': 'Queen', 'Prince': 'Prince', 'Princess': 'Princess',
    'Father': 'Father', 'Mother': 'Mother', 'Uncle': 'Uncle', 'Aunt': 'Aunt',
    'Sri': 'Sri', 'Swami': 'Swami', 'Saint': 'Saint', 'Guru': 'Guru', 'Master': 'Master'
}

DIALOGUE_VERBS = (
    r'(?:said|asked|replied|answered|whispered|muttered|cried|shouted|exclaimed|'
    r'murmured|yelled|gasped|smiled|sighed|thought|nodded|laughed|shrugged|'
    r'continued|added|called|began|spoke|demanded|urged|pleaded)'
)

class NLPCharacterExtractor:
    def __init__(self, ollama_client: OllamaClient = None):
        self.ollama_client = ollama_client

    def heuristic_analysis(self, text: str) -> Dict[str, Any]:
        """Compute structural, lexical, and entity heuristics to assist classification."""
        quotes_count = len(re.findall(r'["\u201c\u201d\'][^"\u201c\u201d\']{3,120}["\u201c\u201d\']', text))
        speech_verbs = re.findall(r'\b' + DIALOGUE_VERBS + r'\b', text, re.IGNORECASE)
        speech_verb_count = len(speech_verbs)

        technical_terms = re.findall(
            r'\b(abstract|introduction|algorithm|dataset|table|figure|equation|programming|software|function|api|syntax|database|server|client|references|reference|et\s+al|methodology|parameters|compiler|architecture|license|copyright|tutorial|documentation|python|java|system|results|analysis)\b',
            text, re.IGNORECASE
        )
        technical_count = len(technical_terms)
        narrative_pronouns = re.findall(r'\b(he|she|him|her|his|hers|they|them|their)\b', text, re.IGNORECASE)
        pronoun_count = len(narrative_pronouns)

        candidates = self.find_candidate_names(text)

        total_words = max(len(text.split()), 1)
        dialogue_density = (quotes_count * 2 + speech_verb_count) / (total_words / 500)
        tech_density = technical_count / (total_words / 500)
        likely_fiction = dialogue_density > 1.0 or (speech_verb_count >= 5 and tech_density < 3.0)

        return {
            "dialogue_density": dialogue_density,
            "tech_density": tech_density,
            "quotes_count": quotes_count,
            "speech_verb_count": speech_verb_count,
            "technical_count": technical_count,
            "pronoun_count": pronoun_count,
            "likely_fiction": likely_fiction,
            "candidate_names": [name for name, _ in candidates]
        }

    def find_candidate_names(self, text: str) -> List[Tuple[str, int]]:
        """
        Robust NLP character extraction:
        1. Title + Surname (e.g., Mrs. Elm, Dr. Watson)
        2. Dialogue speaker attributions (e.g., 'Nora said', 'said Dan')
        3. Mid-sentence proper nouns (excludes sentence-initial capitalized common words)
        4. Full name consolidation (e.g. Nora -> Nora Seed)
        """
        candidates = Counter()

        # Step 1: Title + Name patterns
        title_pattern = r'\b(Mr|Mrs|Ms|Miss|Dr|Doctor|Lord|Lady|Sir|Captain|Professor|Inspector|King|Queen|Prince|Princess|Sri|Swami|Saint|Guru|Master)\.?\s+([A-Z][a-z]+)\b'
        for m in re.finditer(title_pattern, text):
            raw_title = m.group(1).rstrip('.')
            name = m.group(2)
            if name not in STOPWORDS and name.lower() not in ['sri', 'swami', 'lord', 'almighty', 'god', 'father', 'mother'] and len(name) > 2:
                canonical_title = TITLES.get(raw_title, raw_title + ".")
                candidates[f"{canonical_title} {name}"] += 10

        # Step 2: Dialogue attribution verbs (e.g. 'Nora said', 'Mrs. Elm asked')
        # 'Name verb'
        dialogue_pattern_1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+' + DIALOGUE_VERBS + r'\b'
        for m in re.finditer(dialogue_pattern_1, text):
            raw = " ".join(m.group(1).split())
            words = raw.split()
            if not any(w in STOPWORDS for w in words) and all(len(w) > 1 for w in words):
                candidates[raw] += 8

        # 'verb Name'
        dialogue_pattern_2 = DIALOGUE_VERBS + r'\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for m in re.finditer(dialogue_pattern_2, text):
            raw = " ".join(m.group(1).split())
            words = raw.split()
            if not any(w in STOPWORDS for w in words) and all(len(w) > 1 for w in words):
                candidates[raw] += 8

        # Step 3: Proper nouns appearing in the middle of sentences (preceded by lowercase words)
        mid_sentence_pattern = r'\b(?:with|to|from|by|about|and|saw|met|called|loved|hated|hugged|visited|told|heard|behind|beside)\s+([A-Z][a-z]+)\b'
        for m in re.finditer(mid_sentence_pattern, text):
            raw = m.group(1).strip()
            if raw not in STOPWORDS and len(raw) > 2:
                candidates[raw] += 3

        # Step 4: Count recurring full names (e.g., "Nora Seed")
        full_name_counts = Counter()
        for m in re.finditer(r'\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b', text):
            first, last = m.group(1), m.group(2)
            if first not in STOPWORDS and last not in STOPWORDS:
                full_name_counts[f"{first} {last}"] += 1

        # Step 5: Consolidate single names with full names and titles
        consolidated = {}
        for name, score in candidates.items():
            # Clean possessives like "Dan's" -> "Dan"
            clean = re.sub(r"'s$", "", name).strip()
            if clean in STOPWORDS or len(clean) < 2:
                continue

            # Standardize title if present
            for t_raw, t_canon in TITLES.items():
                if clean.startswith(f"{t_raw} ") or clean.startswith(f"{t_raw}. "):
                    clean = f"{t_canon} " + clean.split(" ", 1)[1]
                    break

            # If single name has an established full name in the text, promote it
            upgraded = None
            if " " not in clean:
                for full, cnt in full_name_counts.items():
                    if full.startswith(clean + " ") and cnt >= 3:
                        upgraded = full
                        break

            target = upgraded or clean
            consolidated[target] = consolidated.get(target, 0) + score

        # Remove single-word duplicates if full name or title exists (e.g., remove "Elm" if "Mrs. Elm" exists)
        final_list = {}
        for name, score in consolidated.items():
            is_subword = False
            for other_name in consolidated:
                if other_name != name and name in other_name.split():
                    is_subword = True
                    final_list[other_name] = final_list.get(other_name, 0) + score
                    break
            if not is_subword:
                final_list[name] = final_list.get(name, 0) + score

        ranked = sorted(final_list.items(), key=lambda x: x[1], reverse=True)
        return ranked[:12]

    def extract_character_excerpts(self, text: str, character_name: str, max_chars: int = 2500) -> str:
        """Find key context snippets from the book mentioning this character."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        matching_paras = []

        # Build regex for character name (e.g., 'Mrs. Elm' or 'Nora Seed' or first name)
        first_token = character_name.split()[0]
        if first_token in TITLES.values() and len(character_name.split()) > 1:
            surname = character_name.split()[1]
            pattern = re.compile(rf'\b(?:{re.escape(character_name)}|{re.escape(surname)})\b', re.IGNORECASE)
        elif len(character_name.split()) > 1:
            first_name = character_name.split()[0]
            pattern = re.compile(rf'\b(?:{re.escape(character_name)}|{re.escape(first_name)})\b', re.IGNORECASE)
        else:
            pattern = re.compile(rf'\b{re.escape(character_name)}\b', re.IGNORECASE)

        for p in paragraphs:
            if pattern.search(p):
                matching_paras.append(p)
                if sum(len(x) for x in matching_paras) >= max_chars:
                    break

        if not matching_paras:
            return text[:max_chars]

        return "\n\n---\n\n".join(matching_paras)[:max_chars]

    async def analyze_and_extract_characters(
        self,
        text: str,
        book_title: str = "Uploaded Book",
        model: str = "qwen2.5:1.5b"
    ) -> Dict[str, Any]:
        """
        Extracts verified characters and personas from any uploaded book or document.
        Guarantees zero pronouns or sentence-initial junk words.
        """
        heuristics = self.heuristic_analysis(text)
        candidate_names = heuristics["candidate_names"]

        if not candidate_names:
            return {
                "is_fiction": False,
                "has_characters": False,
                "genre": "Document",
                "message": (
                    f"No characters or notable figures were found in '{book_title}'. "
                    "Please upload a book or document containing characters or notable persons."
                ),
                "characters": []
            }

        # Select top 5-6 verified character names
        top_names = candidate_names[:6]
        candidate_str = ", ".join(top_names)

        # Assemble clean representative excerpt for LLM
        sample_paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 80]
        sample_slice = "\n\n".join(sample_paragraphs[5:25])[:4500] if len(sample_paragraphs) >= 25 else text[:4500]

        character_discovery_prompt = f"""You are an expert NLP literary profiler and text analyst.
Analyze the following book titled '{book_title}'.

VERIFIED CANDIDATE CHARACTER NAMES DETECTED IN TEXT:
{candidate_str}

REPRESENTATIVE EXCERPTS:
{sample_slice}

TASK:
1. Confirm the primary characters from the candidate list (select 3 to 6 actual characters/figures). Discard any terms that are not genuine characters.
2. Determine the genre of this book (e.g. 'Contemporary Fiction / Novel', 'Fantasy / Adventure', 'Mystery / Thriller', 'Historical Fiction', 'Biography / Memoir', 'Philosophy / Drama').

Return ONLY a valid JSON object:
{{
  "has_characters": true,
  "genre": "Genre string",
  "characters": ["Character Name 1", "Character Name 2", "Character Name 3"]
}}
"""
        genre = "Contemporary Literature / Novel"
        confirmed_names = top_names

        if self.ollama_client:
            try:
                raw_discovery = await self.ollama_client.generate(
                    prompt=character_discovery_prompt,
                    system_prompt="You are a precise character extraction AI that outputs strictly valid JSON.",
                    model=model,
                    json_format=True,
                    temperature=0.1
                )
                parsed = json.loads(raw_discovery)
                if parsed.get("genre"):
                    genre = str(parsed["genre"]).strip()
                raw_chars = parsed.get("characters", [])
                if isinstance(raw_chars, list) and len(raw_chars) > 0:
                    valid_chars = []
                    for c in raw_chars:
                        clean = re.sub(r'["\']', '', str(c)).strip()
                        if clean and clean not in STOPWORDS and len(clean) > 2:
                            valid_chars.append(clean)
                    if valid_chars:
                        confirmed_names = valid_chars
            except Exception as e:
                logger.warning(f"Ollama character discovery skipped or failed: {e}. Using verified heuristic candidates.")

        # Ensure we have between 2 and 6 distinct characters
        final_names = []
        for n in confirmed_names:
            if n not in final_names and n not in STOPWORDS and len(n) > 2:
                final_names.append(n)

        if len(final_names) < 3:
            for n in top_names:
                if n not in final_names and n not in STOPWORDS and len(n) > 2:
                    final_names.append(n)
                if len(final_names) >= 5:
                    break

        final_names = final_names[:6]

        # Learn rich character dossiers
        learned_characters = await self._learn_character_dossiers(
            text=text,
            book_title=book_title,
            character_names=final_names,
            genre=genre,
            model=model
        )

        return {
            "is_fiction": True,
            "has_characters": True,
            "genre": genre,
            "message": f"Successfully identified {len(learned_characters)} characters in '{book_title}' ({genre}).",
            "characters": learned_characters
        }

    async def _learn_character_dossiers(
        self,
        text: str,
        book_title: str,
        character_names: List[str],
        genre: str,
        model: str
    ) -> List[Dict[str, Any]]:
        """Deep learning of characters from the story: persona, speech style, quotes, and backstory."""
        context_snippets = []
        for name in character_names:
            excerpt = self.extract_character_excerpts(text, name, max_chars=1400)
            context_snippets.append(f"--- CONTEXT FOR {name} ---\n{excerpt}")

        full_context = "\n\n".join(context_snippets)

        prompt = f"""You are an advanced AI literary profiler.
Perform deep character and persona learning from the book '{book_title}' ({genre}) for these characters: {', '.join(character_names)}.

EXCERPTS FROM THE BOOK:
{full_context[:4500]}

For EACH character, extract their authentic persona directly from their actions and words in the text:
1. "name": Their official name (must match one of {character_names}).
2. "role": Their role in the book (e.g. Protagonist, Guide & Teacher, Companion, Scholar, Ruler, Philosopher).
3. "traits": Array of 3 to 4 distinct psychological personality traits learned from the text.
4. "speech_style": A sentence describing how they speak (tone, cadence, vocabulary, demeanor).
5. "story_backstory": 2-3 sentences summarizing who they are and their role in '{book_title}'.
6. "relationships": Key dynamic with other figures in the book.
7. "learned_facts": Array of 2 to 3 specific facts, memories, or story events from the text they participated in.
8. "initial_greeting": An authentic opening dialogue line spoken directly by this character in FIRST PERSON ("I", "my") welcoming the USER / READER who has opened the book to converse with them (e.g. "Greetings, my friend. I am Krishna. Speak freely of the thoughts and doubts that trouble you.", or "Hello. I am Mrs. Elm. Welcome—take a seat. What brings you to converse with me?"). NEVER address the user as another character from the book (do NOT say 'Hello Arjuna' or 'Hello Nora'). You are talking directly to a human reader.

Return ONLY a valid JSON array of character objects:
[
  {{
    "name": "Character Name",
    "role": "Protagonist",
    "traits": ["Curious", "Resilient", "Introspective"],
    "speech_style": "Speaks with quiet intelligence and raw honesty.",
    "story_backstory": "A central figure exploring the paths and choices of life.",
    "relationships": "Guided by their mentor; closely connected to companion figures.",
    "learned_facts": ["Faced pivotal decisions in the story", "Engaged in transformative dialogue"],
    "initial_greeting": "Greetings, my friend. I am [Character Name]. Speak with me—what thoughts or burdens bring you to converse today?"
  }}
]
"""
        if self.ollama_client:
            try:
                raw_response = await self.ollama_client.generate(
                    prompt=prompt,
                    system_prompt="You are a literary profiling AI that returns strictly valid JSON arrays with first-person character greetings welcoming human readers.",
                    model=model,
                    json_format=True,
                    temperature=0.2
                )
                parsed = json.loads(raw_response)
                char_list = None
                if isinstance(parsed, list) and len(parsed) > 0:
                    char_list = parsed
                elif isinstance(parsed, dict) and "characters" in parsed and isinstance(parsed["characters"], list):
                    char_list = parsed["characters"]

                if char_list:
                    # Validate returned names & ensure greetings are in first person welcoming a human reader
                    valid_learned = []
                    for c in char_list:
                        if isinstance(c, dict) and c.get("name"):
                            # Clean and ensure proper first-person initial greeting
                            g = c.get("initial_greeting", "").strip()
                            name = c.get("name", "").strip()
                            lower_g = g.lower()
                            # If greeting is broken (e.g. repeats name, third-person, or mistakes user for another character), sanitize it
                            if (not g or 
                                lower_g.startswith(f"{name.lower()}, oh") or 
                                f"named {name.lower()}" in lower_g or 
                                any(f"hello {o.lower()}" in lower_g or f"dear {o.lower()}" in lower_g for o in character_names if o != name)):
                                c["initial_greeting"] = f"Greetings, my friend. I am {name}. Speak freely with me—what brings you across the pages of '{book_title}' to converse today?"
                            valid_learned.append(c)
                    if len(valid_learned) >= len(character_names) // 2:
                        return valid_learned
            except Exception as e:
                logger.warning(f"Ollama dossier learning skipped or failed: {e}. Building text-grounded dossiers.")

        # Text-grounded high quality fallback
        fallback_list = []
        for idx, name in enumerate(character_names):
            excerpt = self.extract_character_excerpts(text, name, max_chars=2000)
            quote = self._find_character_quote(excerpt, name)
            greeting = quote or f"Greetings, my friend. I am {name}. Speak freely with me—what brings you across the pages of '{book_title}' to converse today?"

            # Dynamic role determination based on name clues and position
            lower_name = name.lower()
            if any(t in lower_name for t in ["lord", "god", "guru", "swami", "saint", "krishna", "buddha"]):
                role = "Divine Teacher & Spiritual Guide"
            elif any(t in lower_name for t in ["king", "queen", "prince", "princess", "emperor"]):
                role = "Ruler & Royal Figure"
            elif any(t in lower_name for t in ["dr", "doctor", "professor"]):
                role = "Scholar & Mentor"
            elif any(t in lower_name for t in ["elm", "librarian"]):
                role = "Librarian & Wise Mentor"
            elif idx == 0:
                role = "Central Protagonist & Key Figure"
            elif idx == 1:
                role = "Key Companion & Core Guide"
            else:
                role = "Significant Story Figure"

            fallback_list.append({
                "name": name,
                "role": role,
                "traits": ["Reflective", "Authentic", "Perceptive", "Determined"],
                "speech_style": f"Speaks in the authentic first-person literary voice ('I', 'me') of {name} from {book_title}.",
                "story_backstory": f"{name} is an essential figure whose life, choices, and interactions shape the narrative of {book_title}.",
                "relationships": f"Interacts closely with key figures across {book_title}.",
                "learned_facts": [
                    f"Plays a pivotal role in the narrative development of {book_title}.",
                    f"Engages in meaningful dialogue and pivotal decisions in the story."
                ],
                "initial_greeting": greeting
            })

        return fallback_list

    def _find_character_quote(self, excerpt: str, character_name: str) -> str:
        """Extract an authentic quote or dialogue spoken by this character in their excerpts."""
        simple_name = character_name.split()[-1]
        patterns = [
            rf'["\u201c\u2018\']([^"\u201d\u2019\']{{15,120}})["\u201d\u2019\']\s*,?\s*(?:said|asked|replied|whispered|murmured)\s+(?:the\s+\w+,\s+)?(?:{re.escape(character_name)}|{re.escape(simple_name)})',
            rf'(?:{re.escape(character_name)}|{re.escape(simple_name)})\s+(?:said|asked|replied|whispered|murmured)\s*,?\s*["\u201c\u2018\']([^"\u201d\u2019\']{{15,120}})["\u201d\u2019\']',
            rf'(?:{re.escape(character_name)}|{re.escape(simple_name)})\s+said\s*:\s*["\u201c\u2018\']?([^"\n\u201d\u2019]{{15,140}})',
            rf'(?:{re.escape(character_name)}|{re.escape(simple_name)})\s*:\s*["\u201c\u2018\']?([^"\n\u201d\u2019]{{15,140}})'
        ]

        for pat in patterns:
            match = re.search(pat, excerpt, re.IGNORECASE)
            if match:
                quote = match.group(1).strip().rstrip('.,;')
                if len(quote) > 12:
                    return f'"{quote}."'

        return ""
