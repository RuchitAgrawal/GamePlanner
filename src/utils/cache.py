"""
Disk-backed cache for LLM-generated explanations.

Explanations are expensive to generate (LLM call) and deterministic for a given
(user_id, item_id) pair, so we generate each one once and cache it permanently.
This keeps Gemini free-tier usage manageable even as the API gets more traffic.

Storage: a flat JSON file. Simple and inspectable. SQLite is a better choice
if this scales to thousands of users, but JSON works fine for a demo.
"""

import json
import logging
from pathlib import Path

from src.utils.config import CACHE_PATH

log = logging.getLogger(__name__)


class ExplanationCache:
    """
    Read/write cache for (user_id, item_id) -> explanation strings.

    Loads all cached explanations into memory on init for fast lookups,
    and writes back to disk on every set() call.
    """

    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = Path(cache_path)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            log.info("Loaded explanation cache: %d entries", len(self._data))
        else:
            log.info("No existing cache found, starting fresh")

    def _save(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def _key(self, user_id: str, item_id: str) -> str:
        return f"{user_id}|{item_id}"

    def get(self, user_id: str, item_id: str) -> str | None:
        """Return cached explanation or None if not found."""
        return self._data.get(self._key(user_id, item_id))

    def set(self, user_id: str, item_id: str, explanation: str) -> None:
        """Store an explanation and persist to disk."""
        self._data[self._key(user_id, item_id)] = explanation
        self._save()

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        """Wipe the cache (useful for testing)."""
        self._data = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
