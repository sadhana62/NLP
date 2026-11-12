import numpy as np
import re
import os
from typing import List, Dict, Any
import logging
import hashlib

logger = logging.getLogger("pageecho.rag")

class RAGService:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vocab: Dict[str, int] = {}
        self.tfidf_matrix: Any = None
        self.idf: Any = None
        self._index_cache: Dict[str, tuple] = {}

    def tokenize(self, text: str) -> List[str]:
        """Simple, fast tokenization for TF-IDF vectorizer."""
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

    def index_book(self, chunks: List[Dict[str, Any]]) -> int:
        """Build TF-IDF Vector Index over book chunks."""
        fingerprint = hashlib.sha256(
            "\n\n".join(chunk.get("text", "") for chunk in chunks).encode("utf-8")
        ).hexdigest()

        cached = self._index_cache.get(fingerprint)
        if cached is not None:
            self.chunks, self.vocab, self.tfidf_matrix, self.idf = cached
            logger.info("Reused cached index for %d chunks", len(self.chunks))
            return len(self.chunks)

        self.chunks = chunks
        if not chunks:
            self.vocab = {}
            self.tfidf_matrix = None
            self.idf = None
            return 0

        # Build Vocabulary
        doc_tokens = [self.tokenize(c["text"]) for c in chunks]
        vocab_set = set()
        for tokens in doc_tokens:
            vocab_set.update(tokens)

        self.vocab = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocab)
        num_docs = len(chunks)

        if vocab_size == 0 or num_docs == 0:
            self.vocab = {}
            self.tfidf_matrix = None
            self.idf = None
            return 0

        # Term Frequencies & Document Frequencies
        tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
        df = np.zeros(vocab_size, dtype=np.float32)

        for doc_idx, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            for t in tokens:
                if t in self.vocab:
                    term_idx = self.vocab[t]
                    tf_matrix[doc_idx, term_idx] += 1.0

            # Normalize TF by document length
            doc_len = len(tokens)
            if doc_len > 0:
                tf_matrix[doc_idx] /= doc_len

            # Track Document Frequency
            unique_terms = set(tokens)
            for t in unique_terms:
                if t in self.vocab:
                    df[self.vocab[t]] += 1.0

        # Inverse Document Frequency (Smooth IDF)
        self.idf = np.log((1.0 + num_docs) / (1.0 + df)) + 1.0
        self.tfidf_matrix = tf_matrix * self.idf

        # L2 Normalize vectors for Cosine Similarity
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix /= norms

        self._index_cache[fingerprint] = (
            self.chunks, self.vocab, self.tfidf_matrix, self.idf
        )

        logger.info(f"Indexed {num_docs} chunks with vocabulary size {vocab_size}")
        return num_docs

    def save_index_to_disk(self, cache_path: str):
        """Persist computed TF-IDF matrix, idf, and vocabulary to disk."""
        if self.tfidf_matrix is not None and self.vocab and self.idf is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            vocab_items = np.array(list(self.vocab.items()), dtype=object)
            np.savez_compressed(
                cache_path,
                tfidf_matrix=self.tfidf_matrix,
                idf=self.idf,
                vocab=vocab_items
            )
            logger.info("Saved vector index to %s", cache_path)

    def load_index_from_disk(self, cache_path: str, chunks: List[Dict[str, Any]]) -> bool:
        """Load pre-computed TF-IDF matrix, idf, and vocabulary from disk without recalculating."""
        if not os.path.exists(cache_path):
            return False
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                self.tfidf_matrix = np.array(data["tfidf_matrix"])
                self.idf = np.array(data["idf"])
                self.vocab = dict(data["vocab"])
            self.chunks = chunks
            fingerprint = hashlib.sha256(
                "\n\n".join(chunk.get("text", "") for chunk in chunks).encode("utf-8")
            ).hexdigest()
            self._index_cache[fingerprint] = (
                self.chunks, self.vocab, self.tfidf_matrix, self.idf
            )
            logger.info("Loaded pre-computed vector index from disk: %s (%d chunks, %d vocab)", cache_path, len(chunks), len(self.vocab))
            return True
        except Exception as e:
            logger.warning(f"Failed to load vector index from {cache_path}: {e}")
            return False

    def retrieve(self, query: str, character_name: str = "", top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top_k most relevant passages for a query and character."""
        if not self.chunks or self.tfidf_matrix is None or not self.vocab:
            return []

        search_text = f"{character_name} {query}" if character_name else query
        query_tokens = self.tokenize(search_text)
        vocab_size = len(self.vocab)

        query_tf = np.zeros(vocab_size, dtype=np.float32)
        for t in query_tokens:
            if t in self.vocab:
                query_tf[self.vocab[t]] += 1.0

        q_len = len(query_tokens)
        if q_len > 0:
            query_tf /= q_len

        query_tfidf = query_tf * self.idf
        q_norm = np.linalg.norm(query_tfidf)
        if q_norm > 0:
            query_tfidf /= q_norm

        # Cosine similarity matrix multiplication
        similarities = np.dot(self.tfidf_matrix, query_tfidf)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.01:  # Only include non-zero relevance
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["relevance_score"] = round(score, 4)
                results.append(chunk_copy)

        return results
