"""
Интеграционный тест UniversalLLMClient с logprobs.
"""
import os
import pytest
from dotenv import load_dotenv

from kraken_llm import LLMConfig, create_universal_client


def _env_ready() -> bool:
    return bool(os.path.exists(".env"))


@pytest.mark.asyncio
async def test_universal_include_confidence_live():
    if not _env_ready():
        pytest.skip(".env не найден")
    load_dotenv(".env")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:8080"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL", "chat"),
    )

    async with create_universal_client(config) as client:
        result = await client.chat_completion(
            messages=[{"role": "user", "content": "Объясни, что такое ИИ простыми словами"}],
            include_confidence=True,
            max_tokens=128,
        )
        assert isinstance(result, dict)
        assert 0.0 <= result["confidence"] <= 1.0
        assert "text" in result
