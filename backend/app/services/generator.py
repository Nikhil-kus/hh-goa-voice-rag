"""
Modular LLM generation service.

Provider selected via LLM_PROVIDER env var.
Supported: groq | openai

The prompt instructs the model to answer ONLY from the provided passages.
<think> blocks (chain-of-thought) are stripped from the final answer.
"""
from __future__ import annotations

import asyncio
import re
from typing import List, Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the information in the provided passages.
- If the passages do not contain enough information, say exactly: "I don't have sufficient information in the provided knowledge base to answer that."
- Do NOT use any external knowledge.
- Keep your answer concise and factual (2-4 sentences max).
- Do NOT make up information.
- Do NOT include any preamble or reasoning — output only the final answer."""

CONTEXT_TEMPLATE = """Passages:
{passages}

Question: {question}

Answer:"""


def _build_prompt(question: str, passages: List[str]) -> str:
    formatted = "\n\n".join(
        f"[Passage {i+1}] {p}" for i, p in enumerate(passages)
    )
    return CONTEXT_TEMPLATE.format(passages=formatted, question=question)


REFUSAL_PHRASE = "I don't have sufficient information in the provided knowledge base to answer that."

def _clean_answer(text: str) -> str:
    """
    Strip <think>...</think> chain-of-thought blocks produced by some models
    (e.g. Qwen3). Return only the final answer text.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    return text


def _is_llm_refusal(text: str) -> bool:
    """Return True if the LLM itself returned the standard refusal phrase."""
    return "sufficient information" in text.lower() and "knowledge base" in text.lower()


async def generate_answer(
    question: str,
    passages: List[str],
    timeout_ms: int,
) -> Optional[str]:
    """
    Call the configured LLM provider and return the generated answer.
    Returns None on failure (caller handles refusal).
    """
    provider = settings.llm_provider.lower()
    try:
        if provider == "groq":
            return await _generate_groq(question, passages, timeout_ms)
        elif provider == "openai":
            return await _generate_openai(question, passages, timeout_ms)
        else:
            logger.error("Unknown LLM provider", extra={"provider": provider})
            return None
    except asyncio.TimeoutError:
        logger.warning("LLM generation timed out",
                       extra={"provider": provider, "timeout_ms": timeout_ms})
        return None
    except Exception as e:
        logger.error("LLM generation failed",
                     extra={"provider": provider, "error": str(e)})
        return None


async def _generate_groq(question: str, passages: List[str], timeout_ms: int) -> Optional[str]:
    if not settings.groq_api_key:
        logger.error("GROQ_API_KEY not set")
        return None

    from groq import AsyncGroq
    client = AsyncGroq(api_key=settings.groq_api_key)
    user_content = _build_prompt(question, passages)

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.1,
        ),
        timeout=timeout_ms / 1000,
    )
    raw = response.choices[0].message.content or ""
    answer = _clean_answer(raw)
    return answer if answer else None


async def _generate_openai(question: str, passages: List[str], timeout_ms: int) -> Optional[str]:
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not set")
        return None

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    user_content = _build_prompt(question, passages)

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=256,
            temperature=0.1,
        ),
        timeout=timeout_ms / 1000,
    )
    raw = response.choices[0].message.content or ""
    answer = _clean_answer(raw)
    return answer if answer else None
