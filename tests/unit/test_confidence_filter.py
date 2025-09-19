"""
Интеграционные тесты фильтрации и перегенерации по уверенности с реальными сетевыми вызовами.
Требуется .env в корне.
"""
import os
import pytest

from dotenv import load_dotenv
from kraken_llm import LLMConfig, create_standard_client
from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat


def _env_ready() -> bool:
    return bool(os.path.exists(".env"))


@pytest.mark.asyncio
async def test_ensure_confident_chat_live_standard():
    if not _env_ready():
        pytest.skip(".env не найден")
    load_dotenv(".env")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:8080"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL", "chat"),
    )

    async with create_standard_client(config) as client:
        cfg = ConfidenceFilterConfig(
            min_confidence=0.6,
            max_attempts=2,
            prefer_streaming=True,
            per_token_threshold=0.3,
            max_low_conf_fraction=0.5,
        )
        result = await ensure_confident_chat(
            client,
            messages=[{"role": "user", "content": "Кратко объясни ИИ простыми словами"}],
            cfg=cfg,
            max_tokens=128,
        )
        assert isinstance(result, dict)
        assert "text" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["attempts_made"] >= 1
