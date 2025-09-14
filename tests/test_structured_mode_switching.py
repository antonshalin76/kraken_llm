"""
Тесты для переключения режимов structured output между OpenAI и Outlines.

Этот модуль содержит тесты для проверки корректности работы переключателя
outlines_so_mode между нативным OpenAI structured output и Outlines.
"""

import asyncio
import pytest
import os
from typing import List, Optional
from pydantic import BaseModel, Field

from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError

from dotenv import load_dotenv

load_dotenv()


class SimpleResponseModel(BaseModel):
    """Тестовая модель для structured output."""
    message: str = Field(description="Основное сообщение")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Уровень уверенности")
    tags: List[str] = Field(description="Список тегов")
    metadata: Optional[dict] = Field(
        default=None, description="Дополнительные метаданные")


class ComplexResponseModel(BaseModel):
    """Сложная модель для тестирования."""
    title: str = Field(description="Заголовок")
    items: List[SimpleResponseModel] = Field(description="Список элементов")
    total_count: int = Field(ge=0, description="Общее количество")
    is_complete: bool = Field(description="Флаг завершенности")


@pytest.fixture
def openai_config():
    """Конфигурация для нативного OpenAI режима."""
    return LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.1,
        max_tokens=1000,
        outlines_so_mode=False  # Нативный OpenAI режим
    )


@pytest.fixture
def outlines_config():
    """Конфигурация для Outlines режима."""
    return LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.1,
        max_tokens=1000,
        outlines_so_mode=True  # Outlines режим
    )


@pytest.fixture
def test_messages():
    """Тестовые сообщения для structured output."""
    return [
        {
            "role": "system",
            "content": "Ты помощник, который отвечает в структурированном формате JSON."
        },
        {
            "role": "user",
            "content": "Создай тестовое сообщение с уверенностью 0.9 и тегами ['test', 'example']"
        }
    ]


class TestStructuredModeSwitching:
    """Тесты переключения режимов structured output."""

    @pytest.mark.asyncio
    async def test_openai_mode_initialization(self, openai_config):
        """Тест инициализации клиента в режиме OpenAI."""
        client = StructuredLLMClient(openai_config)

        # Проверяем, что режим установлен правильно
        assert client.config.outlines_so_mode is False
        assert client.validator is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_outlines_mode_initialization(self, outlines_config):
        """Тест инициализации клиента в режиме Outlines."""
        client = StructuredLLMClient(outlines_config)

        # Проверяем, что режим установлен правильно
        assert client.config.outlines_so_mode is True
        assert client.validator is not None

        await client.close()

    @pytest.mark.asyncio
    async def test_mode_switching_validation(self, openai_config, outlines_config, test_messages):
        """Тест валидации входных параметров в обоих режимах."""

        # Тест OpenAI режима
        openai_client = StructuredLLMClient(openai_config)

        # Проверяем валидацию пустых сообщений
        with pytest.raises(ValidationError, match="Список сообщений не может быть пустым"):
            await openai_client.structured_completion([], SimpleResponseModel)

        # Проверяем валидацию отсутствующей модели
        with pytest.raises(ValidationError, match="response_model обязателен для structured output"):
            await openai_client.structured_completion(test_messages, None)

        await openai_client.close()

        # Тест Outlines режима
        outlines_client = StructuredLLMClient(outlines_config)

        # Проверяем валидацию пустых сообщений
        with pytest.raises(ValidationError, match="Список сообщений не может быть пустым"):
            await outlines_client.structured_completion([], SimpleResponseModel)

        # Проверяем валидацию отсутствующей модели
        with pytest.raises(ValidationError, match="response_model обязателен для structured output"):
            await outlines_client.structured_completion(test_messages, None)

        await outlines_client.close()

    @pytest.mark.asyncio
    async def test_method_routing_openai_mode(self, openai_config, test_messages, monkeypatch):
        """Тест маршрутизации методов в OpenAI режиме."""
        client = StructuredLLMClient(openai_config)

        # Мокаем методы для проверки вызовов
        openai_called = False
        outlines_called = False

        async def mock_openai_method(*args, **kwargs):
            nonlocal openai_called
            openai_called = True
            # Возвращаем мок-объект
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        async def mock_outlines_method(*args, **kwargs):
            nonlocal outlines_called
            outlines_called = True
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        # Подменяем методы
        monkeypatch.setattr(
            client, "_structured_non_stream_openai", mock_openai_method)
        monkeypatch.setattr(
            client, "_structured_non_stream_outlines", mock_outlines_method)

        # Вызываем non-streaming метод
        result = await client.structured_completion(test_messages, SimpleResponseModel, stream=False)

        # Проверяем, что вызван правильный метод
        assert openai_called is True
        assert outlines_called is False
        assert isinstance(result, SimpleResponseModel)

        await client.close()

    @pytest.mark.asyncio
    async def test_method_routing_outlines_mode(self, outlines_config, test_messages, monkeypatch):
        """Тест маршрутизации методов в Outlines режиме."""
        client = StructuredLLMClient(outlines_config)

        # Мокаем методы для проверки вызовов
        openai_called = False
        outlines_called = False

        async def mock_openai_method(*args, **kwargs):
            nonlocal openai_called
            openai_called = True
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        async def mock_outlines_method(*args, **kwargs):
            nonlocal outlines_called
            outlines_called = True
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        # Подменяем методы
        monkeypatch.setattr(
            client, "_structured_non_stream_openai", mock_openai_method)
        monkeypatch.setattr(
            client, "_structured_non_stream_outlines", mock_outlines_method)

        # Вызываем non-streaming метод
        result = await client.structured_completion(test_messages, SimpleResponseModel, stream=False)

        # Проверяем, что вызван правильный метод
        assert openai_called is False
        assert outlines_called is True
        assert isinstance(result, SimpleResponseModel)

        await client.close()

    @pytest.mark.asyncio
    async def test_streaming_mode_routing(self, openai_config, outlines_config, test_messages, monkeypatch):
        """Тест маршрутизации streaming методов."""

        # Тест OpenAI streaming режима
        openai_client = StructuredLLMClient(openai_config)

        openai_stream_called = False
        outlines_non_stream_called = False

        async def mock_openai_stream(*args, **kwargs):
            nonlocal openai_stream_called
            openai_stream_called = True
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        async def mock_outlines_stream(*args, **kwargs):
            nonlocal outlines_non_stream_called
            outlines_non_stream_called = True
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        # Подменяем методы
        monkeypatch.setattr(
            openai_client, "_structured_stream_openai", mock_openai_stream)
        monkeypatch.setattr(
            openai_client, "_structured_stream_outlines", mock_outlines_stream)

        # Вызываем streaming метод в OpenAI режиме
        result = await openai_client.structured_completion(test_messages, SimpleResponseModel, stream=True)

        # Проверяем, что вызван правильный метод
        assert openai_stream_called is True
        assert outlines_non_stream_called is False

        await openai_client.close()

        # Тест Outlines streaming режима (должен переключиться на non-streaming)
        outlines_client = StructuredLLMClient(outlines_config)

        openai_stream_called = False
        outlines_non_stream_called = False

        # Подменяем методы
        monkeypatch.setattr(
            outlines_client, "_structured_stream_openai", mock_openai_stream)
        monkeypatch.setattr(
            outlines_client, "_structured_stream_outlines", mock_outlines_stream)

        # Вызываем streaming метод в Outlines режиме
        result = await outlines_client.structured_completion(test_messages, SimpleResponseModel, stream=True)

        # Проверяем, что вызван outlines stream метод (который внутри делает non-streaming)
        assert openai_stream_called is False
        assert outlines_non_stream_called is True

        await outlines_client.close()

    @pytest.mark.asyncio
    async def test_complex_model_both_modes(self, openai_config, outlines_config, monkeypatch):
        """Тест работы со сложными моделями в обоих режимах."""

        # Создаем сложные тестовые сообщения
        complex_messages = [
            {
                "role": "system",
                "content": "Создай структурированный ответ с заголовком и списком элементов."
            },
            {
                "role": "user",
                "content": "Создай ответ с заголовком 'Test' и двумя элементами"
            }
        ]

        # Мок-результат
        mock_result = ComplexResponseModel(
            title="Test",
            items=[
                SimpleResponseModel(
                    message="item1", confidence=0.8, tags=["tag1"]),
                SimpleResponseModel(
                    message="item2", confidence=0.9, tags=["tag2"])
            ],
            total_count=2,
            is_complete=True
        )

        # Тест OpenAI режима
        openai_client = StructuredLLMClient(openai_config)

        async def mock_openai_complex(*args, **kwargs):
            return mock_result

        monkeypatch.setattr(
            openai_client, "_structured_non_stream_openai", mock_openai_complex)

        result = await openai_client.structured_completion(complex_messages, ComplexResponseModel)
        assert isinstance(result, ComplexResponseModel)
        assert result.title == "Test"
        assert len(result.items) == 2

        await openai_client.close()

        # Тест Outlines режима
        outlines_client = StructuredLLMClient(outlines_config)

        async def mock_outlines_complex(*args, **kwargs):
            return mock_result

        monkeypatch.setattr(
            outlines_client, "_structured_non_stream_outlines", mock_outlines_complex)

        result = await outlines_client.structured_completion(complex_messages, ComplexResponseModel)
        assert isinstance(result, ComplexResponseModel)
        assert result.title == "Test"
        assert len(result.items) == 2

        await outlines_client.close()

    @pytest.mark.asyncio
    async def test_config_parameter_passing(self, openai_config, outlines_config, test_messages, monkeypatch):
        """Тест передачи параметров конфигурации в оба режима."""

        # Тест передачи параметров в OpenAI режим
        openai_client = StructuredLLMClient(openai_config)

        received_params = {}

        async def mock_openai_with_params(messages, response_model, model=None, temperature=None, max_tokens=None, **kwargs):
            received_params["temperature"] = temperature
            received_params["max_tokens"] = max_tokens
            received_params.update(kwargs)
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        monkeypatch.setattr(
            openai_client, "_structured_non_stream", mock_openai_with_params)

        # Вызываем с дополнительными параметрами
        await openai_client.structured_completion(
            test_messages,
            SimpleResponseModel,
            temperature=0.5,
            max_tokens=500
        )

        # Проверяем, что параметры переданы
        assert "temperature" in received_params
        assert "max_tokens" in received_params
        assert received_params["temperature"] == 0.5
        assert received_params["max_tokens"] == 500

        await openai_client.close()

        # Тест передачи параметров в Outlines режим
        outlines_client = StructuredLLMClient(outlines_config)

        received_params = {}

        async def mock_outlines_with_params(messages, response_model, model=None, temperature=None, max_tokens=None, **kwargs):
            received_params["temperature"] = temperature
            received_params["max_tokens"] = max_tokens
            received_params.update(kwargs)
            return SimpleResponseModel(message="test", confidence=0.9, tags=["test"])

        monkeypatch.setattr(
            outlines_client, "_structured_non_stream", mock_outlines_with_params)

        # Вызываем с дополнительными параметрами
        await outlines_client.structured_completion(
            test_messages,
            SimpleResponseModel,
            temperature=0.7,
            max_tokens=800
        )

        # Проверяем, что параметры переданы
        assert "temperature" in received_params
        assert "max_tokens" in received_params
        assert received_params["temperature"] == 0.7
        assert received_params["max_tokens"] == 800

        await outlines_client.close()


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])
