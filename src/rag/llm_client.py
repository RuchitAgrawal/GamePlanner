"""
LLM client for generating explanations and conversational responses.

Primary: Gemini 2.5 Flash Lite (google-genai SDK, free tier, 1000 req/day, 15 req/min).
Fallback: Groq llama-3.1-8b-instant (also free, faster, rate-limited differently).

Keeps a session-level request counter and warns when approaching the daily limit.
Temperature is kept low (0.3) to minimize hallucination in grounded explanations.
"""

import logging
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

_DAILY_LIMIT = 900  # warn at this threshold (Gemini 2.5 Flash Lite free tier is 1000/day)


class LLMClient:
    """
    Wrapper around Gemini 2.0 Flash with optional Groq fallback.

    Usage:
        client = LLMClient()
        text = client.generate(prompt="...", context="...")
    """

    def __init__(
        self,
        gemini_api_key: str | None = None,
        groq_api_key: str | None = None,
        max_tokens: int = 150,
        temperature: float = 0.3,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._request_count = 0

        gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        self._gemini_client = None
        self._groq_client = None

        if gemini_key:
            try:
                from google import genai
                from src.utils.config import GEMINI_MODEL
                self._gemini_client = genai.Client(api_key=gemini_key)
                self._gemini_model = GEMINI_MODEL
                log.info("Gemini %s client ready (1000 req/day, 15 req/min)", GEMINI_MODEL)
            except ImportError:
                log.warning("google-genai not installed. Run: pip install google-genai")

        else:
            log.warning("No GEMINI_API_KEY found. Gemini will not be available.")

        if groq_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=groq_key)
                log.info("Groq fallback client ready")
            except ImportError:
                log.warning("groq package not installed. Groq fallback unavailable.")

        if not self._gemini_client and not self._groq_client:
            log.error("No LLM client configured. Set GEMINI_API_KEY or GROQ_API_KEY in .env")

    @property
    def request_count(self) -> int:
        return self._request_count

    def generate(self, prompt: str, context: str = "") -> str:
        """
        Generate a short text response grounded in the given context.

        Tries Gemini 2.5 Flash Lite first (1000 req/day free).
        Falls back to Groq if Gemini is unavailable or quota is hit.
        Returns an empty string if both fail.
        """
        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt

        # Gemini primary
        if self._gemini_client and self._request_count < _DAILY_LIMIT:
            try:
                result = self._generate_gemini(full_prompt)
                self._request_count += 1
                return result
            except Exception as exc:
                log.warning("Gemini call failed (%s). Trying Groq fallback.", exc)

        # Groq fallback
        if self._groq_client:
            result = self._generate_groq(full_prompt, "")
            if result:
                return result

        log.error("Both LLM clients failed. Returning empty response.")
        return ""

    def _generate_gemini(self, prompt: str) -> str:
        from google import genai
        response = self._gemini_client.models.generate_content(
            model=self._gemini_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        return response.text.strip()

    def _generate_groq(self, prompt: str, context: str) -> str:
        if not self._groq_client:
            return ""
        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
        try:
            completion = self._groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise game recommendation assistant. "
                            "You MUST only reference games explicitly mentioned in the prompt. "
                            "Do NOT invent, assume, or reference any game not listed. "
                            "Write exactly 1-2 sentences."
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            log.error("Groq call failed: %s", exc)
            return ""
