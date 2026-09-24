"""
Grade-scoped retrieval over the CA Common Core math standards
(processed_json_textbooks/, produced by preprocess_textbooks.py). This data
existed already but was never wired into the LLM - the system prompt was
the only "knowledge" it had.

Deliberately lightweight: the corpus is small (9 files, 8-19KB each), so
this uses simple keyword/overlap scoring over sentence-grouped chunks
rather than a real embeddings + vector DB pipeline. That's the right
trade-off for this corpus size on a resource-constrained Pi - if the corpus
grows significantly, real embeddings would scale better, but would add a
new dependency and (for Azure OpenAI embeddings) a new API surface to
verify access for, same class of issue as Speaker Recognition turned out
to be.
"""

import json
import os
import re
import logging
from collections import Counter
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Maps a wide range of natural-language grade answers (as a K-8 student
# might actually say them) to the file-key format used by
# processed_json_textbooks/math_grade_<N>.json / math_kindergarten.json
_GRADE_WORDS = {
    "kindergarten": "K", "kinder": "K",
    "first": "1", "1st": "1", "one": "1",
    "second": "2", "2nd": "2", "two": "2",
    "third": "3", "3rd": "3", "three": "3",
    "fourth": "4", "4th": "4", "four": "4",
    "fifth": "5", "5th": "5", "five": "5",
    "sixth": "6", "6th": "6", "six": "6",
    "seventh": "7", "7th": "7", "seven": "7",
    "eighth": "8", "8th": "8", "eight": "8",
}


def normalize_grade(spoken_text: str) -> Optional[str]:
    """Parse a free-form spoken grade answer ("I'm in third grade", "3rd",
    "grade 5") into 'K' or '1'..'8'. Returns None if nothing recognizable."""
    text = spoken_text.lower().strip()

    match = re.search(r"\b([1-8])\b", text)
    if match:
        return match.group(1)

    for word, grade in _GRADE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return grade

    if "kindergarten" in text or "kinder" in text or text.strip() == "k":
        return "K"

    return None


class TextbookSearch:
    """Loads and chunks the CA Common Core math standards, one set of
    chunks per grade, for lightweight keyword-scored retrieval."""

    _CHUNK_SIZE = 500  # characters per chunk, roughly a paragraph

    def __init__(self, textbooks_dir: str):
        self.textbooks_dir = textbooks_dir
        self._chunks_by_grade: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        grade_files = {
            "K": "math_kindergarten.json",
            **{str(g): f"math_grade_{g}.json" for g in range(1, 9)}
        }
        for grade, filename in grade_files.items():
            path = os.path.join(self.textbooks_dir, filename)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._chunks_by_grade[grade] = self._chunk(data.get("content", ""))
            except Exception as e:
                logger.warning(f"Failed to load textbook content for grade {grade}: {e}")

        total_chunks = sum(len(c) for c in self._chunks_by_grade.values())
        logger.info(f"📚 TextbookSearch loaded {total_chunks} chunks across {len(self._chunks_by_grade)} grades")

    def _chunk(self, text: str) -> List[str]:
        """
        Split into ~_CHUNK_SIZE-character chunks, grouping whole sentences
        together. The source PDFs' single newlines are just line-wrap
        artifacts (confirmed by inspection - they break mid-sentence), not
        paragraph breaks, so whitespace is flattened before splitting on
        sentence boundaries.
        """
        flat = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", flat)

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > self._CHUNK_SIZE and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current.strip())
        return chunks

    @staticmethod
    def _score(query_words: Counter, chunk: str) -> float:
        chunk_words = Counter(re.findall(r"[a-z]+", chunk.lower()))
        overlap = sum(min(count, chunk_words[word]) for word, count in query_words.items())
        return overlap / max(len(chunk_words), 1)

    def search(self, query: str, grade: str, top_k: int = 3) -> List[str]:
        """Returns up to top_k most relevant chunks for `grade`. Empty list
        if the grade is unknown or no chunk scores above zero."""
        chunks = self._chunks_by_grade.get(grade, [])
        if not chunks:
            return []

        query_words = Counter(re.findall(r"[a-z]+", query.lower()))
        scored = [(self._score(query_words, c), c) for c in chunks]
        scored = [(score, c) for score, c in scored if score > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
