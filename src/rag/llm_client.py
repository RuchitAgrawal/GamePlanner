"""
LLM client for generating explanations and conversational responses.

Primary: Gemini 1.5 Flash (free tier, 1500 req/day).
Fallback: Groq (also free, faster, rate-limited differently).

Keeps a session-level request counter and warns when approaching the daily limit.
Temperature is kept low (0.3) to minimize hallucination in grounded explanations.
"""

import logging
import os
from typing import Literal

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

_DAILY_LIMIT = 1400  # warn at this threshold (Gemini free tier is 1500/day)


class LLMClient:
    """
    Wrapper around Gemini 1.5 Flash with optional Groq fallback.

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
            genai.configure(api_key=gemini_key)
            self._gemini_client = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            log.info("Gemini 1.5 Flash client ready")
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

        Tries Gemini first. Falls back to Groq if Gemini is unavailable or quota is hit.
        Returns an empty string if both fail, so callers can handle the fallback gracefully.
        """
        if self._request_count >= _DAILY_LIMIT:
            log.warning("Approaching daily Gemini limit (%d requests). Switching to Groq.", _DAILY_LIMIT)
            return self._generate_groq(prompt, context)

        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt

        if self._gemini_client:
            try:
                result = self._generate_gemini(full_prompt)
                self._request_count += 1
                return result
            except Exception as exc:
                log.warning("Gemini call failed (%s). Trying Groq fallback.", exc)

        if self._groq_client:
            return self._generate_groq(full_prompt, "")

        log.error("Both LLM clients failed. Returning empty response.")
        return ""

    def _generate_gemini(self, prompt: str) -> str:
        response = self._gemini_client.generate_content(prompt)
        return response.text.strip()

    def _generate_groq(self, prompt: str, context: str) -> str:
        if not self._groq_client:
            return ""
        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
        try:
            completion = self._groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            log.error("Groq call failed: %s", exc)
            return ""
