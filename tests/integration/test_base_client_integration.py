"""
Интеграционные тесты для базового LLM клиента.

Тестирует реальное создание и использование BaseLLMClient
с настоящими зависимостями (но без реальных API вызовов).
"""

import pytest
from unittest.mock import AsyncMock, patch

from kraken_llm.client.base import BaseLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.network import NetworkError


class IntegrationLLMClient(BaseLLMClient):
    """Тестовая реализация BaseLLMClient для интеграционных тестов."""

    async def chat_completion(self, messages, **kwargs):
        return "test response"

    async def chat_completion_stream(self, messages, **kwargs):
        yield "chunk1"
        yield "chunk2"

    async def chat_completion_structured(self, messages, response_model, **kwargs):
        return response_model(message="test", confidence=0.9)


class TestBaseLLMClientIntegration:
    """Интеграционные тесты для BaseLLMClient."""

    @pytest.fixture
    def config(self):
        """Конфигурация для интеграционных тестов."""
        return LLMConfig(
            endpoint="http://localhost:8080",
            api_key="test_key",
            model="test_model",
            temperature=0.7,
            max_tokens=1000,
        )

    def test_client_creation_with_real_config(self, config):
        """Тест создания клиента с реальной конфигурацией."""
        # Мокаем только AsyncOpenAI, чтобы не делать реальные запросы
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()

            client = IntegrationLLMClient(config)

            # Проверяем, что клиент создался
            assert client.config == config
            assert client.openai_client is not None

            # Проверяем, что AsyncOpenAI был вызван с правильными параметрами
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args.kwargs

            assert call_kwargs['api_key'] == config.api_key
            assert call_kwargs['base_url'] == f"{config.endpoint}/v1"
            assert call_kwargs['max_retries'] == config.max_retries

            # Проверяем timeout конфигурацию
            timeout = call_kwargs['timeout']
            assert timeout.connect == config.connect_timeout
            assert timeout.read == config.read_timeout
            assert timeout.write == config.write_timeout
            assert timeout.pool is None

    def test_openai_params_preparation(self, config):
        """Тест подготовки параметров для OpenAI API."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()

            client = IntegrationLLMClient(config)

            messages = [{"role": "user", "content": "test"}]

            # Тест с параметрами по умолчанию
            params = client._prepare_openai_params(messages)

            assert params['messages'] == messages
            assert params['model'] == config.model
            assert params['temperature'] == config.temperature
            assert params['max_tokens'] == config.max_tokens
            assert params['top_p'] == config.top_p
            assert params['frequency_penalty'] == config.frequency_penalty
            assert params['presence_penalty'] == config.presence_penalty

            # Тест с переопределениями
            custom_params = client._prepare_openai_params(
                messages,
                model="custom_model",
                temperature=0.9,
                max_tokens=2000,
                stream=True,
                custom_param="value"
            )

            assert custom_params['model'] == "custom_model"
            assert custom_params['temperature'] == 0.9
            assert custom_params['max_tokens'] == 2000
            assert custom_params['stream'] is True
            assert custom_params['custom_param'] == "value"

    @pytest.mark.asyncio
    async def test_client_context_manager(self, config):
        """Тест использования клиента как async context manager."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            async with IntegrationLLMClient(config) as client:
                assert isinstance(client, IntegrationLLMClient)
                assert client.openai_client == mock_client

            # Проверяем, что close был вызван
            mock_client.close.assert_called_once()

    def test_client_repr(self, config):
        """Тест строкового представления клиента."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()

            client = IntegrationLLMClient(config)
            repr_str = repr(client)

            assert "IntegrationLLMClient" in repr_str
            assert config.endpoint in repr_str
            assert config.model in repr_str

    def test_config_validation_integration(self):
        """Тест интеграции валидации конфигурации."""
        # Тест с корректной конфигурацией
        valid_config = LLMConfig(
            endpoint="http://localhost:8080",
            model="test_model"
        )

        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()

            client = IntegrationLLMClient(valid_config)
            assert client.config.endpoint == "http://localhost:8080"
            assert client.config.model == "test_model"

        # Тест с некорректной конфигурацией (Pydantic должен отклонить)
        # Проверим некорректный max_tokens
        with pytest.raises(Exception):
            LLMConfig(
                endpoint="http://localhost:8080",
                model="test_model",
                max_tokens=0  # Некорректное значение
            )
