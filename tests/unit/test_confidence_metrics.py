"""
Интеграционные тесты работы с logprobs через реальные сетевые вызовы.
Требуется настроенный .env в корне проекта.
"""
import os
import pytest

from dotenv import load_dotenv
from kraken_llm import LLMConfig, create_standard_client, create_streaming_client


def _env_ready() -> bool:
    return bool(os.path.exists(".env"))


@pytest.mark.asyncio
async def test_standard_include_confidence_live():
    if not _env_ready():
        pytest.skip(".env не найден")
    load_dotenv(".env")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:8080"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL", "chat"),
    )

    async with create_standard_client(config) as client:
        result = await client.chat_completion(
            messages=[{"role": "user", "content": "Кратко объясни, что такое машинное обучение"}],
            include_confidence=True,
            max_tokens=128,
        )
        assert isinstance(result, dict)
        assert 0.0 <= result["confidence"] <= 1.0
        assert "text" in result and isinstance(result["text"], str)
        assert result["confidence_label"] in {"Очень высокая", "Высокая", "Средняя", "Низкая", "Очень низкая", "Нет logprobs"}


@pytest.mark.asyncio
async def test_streaming_include_confidence_live():
    if not _env_ready():
        pytest.skip(".env не найден")
    load_dotenv(".env")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:8080"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL", "chat"),
    )

    async with create_streaming_client(config) as client:
        result = await client.chat_completion(
            messages=[{"role": "user", "content": "Опиши ИИ простыми словами"}],
            include_confidence=True,
            max_tokens=128,
        )
        assert isinstance(result, dict)
        assert 0.0 <= result["confidence"] <= 1.0
        # В потоковом режиме мы ожидаем пер‑токенные метрики при поддержке сервером
        assert "text" in result and isinstance(result["text"], str)
        assert result["confidence_label"] in {"Очень высокая", "Высокая", "Средняя", "Низкая", "Очень низкая", "Нет logprobs"}
        # Не все провайдеры возвращают токенные logprobs — проверяем опционально
        if "token_confidences" in result:
            assert isinstance(result["token_confidences"], list)
