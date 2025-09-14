"""
Unit тесты для базового абстрактного LLM клиента.

Тестирует функциональность BaseLLMClient, включая инициализацию,
создание клиентов AsyncOpenAI и Outlines, обработку параметров
и утилитарные методы.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from kraken_llm.client.base import BaseLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.api import APIError
from kraken_llm.exceptions.network import NetworkError, TimeoutError
from kraken_llm.exceptions.validation import ValidationError


class ResponseModel(BaseModel):
    """Тестовая Pydantic модель для structured output."""
    message: str
    confidence: float


class ConcreteLLMClient(BaseLLMClient):
    """Конкретная реализация BaseLLMClient для тестирования."""

    async def chat_completion(self, messages, **kwargs):
        return "test response"

    async def chat_completion_stream(self, messages, **kwargs):
        yield "chunk1"
        yield "chunk2"

    async def chat_completion_structured(self, messages, response_model, **kwargs):
        return ResponseModel(message="test", confidence=0.9)


class TestBaseLLMClient:
    """Тесты для BaseLLMClient."""

    @pytest.fixture
    def config(self):
        """Базовая конфигурация для тестов."""
        return LLMConfig(
            endpoint="http://localhost:8080",
            api_key="test_key",
            model="test_model",
            temperature=0.7,
            max_tokens=1000,
        )

    @pytest.fixture
    def mock_client_creation(self):
        """Фикстура для мокирования создания AsyncOpenAI клиента."""
        with patch.object(ConcreteLLMClient, '_create_async_openai_client') as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client
            yield mock_create, mock_client

    @pytest.fixture
    def invalid_config(self):
        """Некорректная конфигурация для тестов ошибок."""
        # Создаем конфигурацию с валидными значениями для Pydantic,
        # но которые будут проверены в _validate_config
        return LLMConfig(
            endpoint="",  # Пустой endpoint
            model="",     # Пустая модель
            max_tokens=100,  # Валидное значение для Pydantic
        )

    def test_init_success(self, config, mock_client_creation):
        """Тест успешной инициализации клиента."""
        mock_create, mock_client = mock_client_creation

        client = ConcreteLLMClient(config)

        assert client.config == config
        assert client.openai_client == mock_client
        assert client._outlines_model is None  # Ленивая инициализация

        # Проверка вызова создания клиента
        mock_create.assert_called_once()

    def test_init_validation_error(self, invalid_config):
        """Тест ошибки валидации при инициализации."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteLLMClient(invalid_config)

        assert "Endpoint не может быть пустым" in str(exc_info.value)

    def test_validate_config_empty_endpoint(self):
        """Тест валидации пустого endpoint."""
        config = LLMConfig(endpoint="", model="test")
        client = ConcreteLLMClient.__new__(ConcreteLLMClient)
        client.config = config

        with pytest.raises(ValidationError) as exc_info:
            client._validate_config()

        assert "Endpoint не может быть пустым" in str(exc_info.value)

    def test_validate_config_empty_model(self):
        """Тест валидации пустой модели."""
        config = LLMConfig(endpoint="http://test", model="")
        client = ConcreteLLMClient.__new__(ConcreteLLMClient)
        client.config = config

        with pytest.raises(ValidationError) as exc_info:
            client._validate_config()

        assert "Model не может быть пустым" in str(exc_info.value)

    def test_validate_config_invalid_max_tokens(self):
        """Тест валидации некорректного max_tokens через Pydantic."""
        with pytest.raises(Exception) as exc_info:
            LLMConfig(endpoint="http://test", model="test", max_tokens=0)

        # Pydantic должен выбросить ValidationError
        assert "greater than 0" in str(exc_info.value)

    def test_validate_config_invalid_temperature(self):
        """Тест валидации некорректной температуры через Pydantic."""
        with pytest.raises(Exception) as exc_info:
            LLMConfig(endpoint="http://test", model="test", temperature=3.0)

        # Pydantic должен выбросить ValidationError
        assert "less than or equal to 2" in str(exc_info.value)

    def test_create_async_openai_client_success(self, config):
        """Тест успешного создания AsyncOpenAI клиента."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = ConcreteLLMClient.__new__(ConcreteLLMClient)
            client.config = config

            result = client._create_async_openai_client()

            assert result == mock_client
            mock_openai.assert_called_once()

            # Проверка параметров
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs['api_key'] == config.api_key
            assert 'base_url' in call_kwargs
            assert call_kwargs['max_retries'] == config.max_retries

            # Проверка timeout конфигурации
            timeout = call_kwargs['timeout']
            assert timeout.connect == config.connect_timeout
            assert timeout.read == config.read_timeout
            assert timeout.write == config.write_timeout

    def test_create_async_openai_client_error(self, config):
        """Тест ошибки создания AsyncOpenAI клиента."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.side_effect = Exception("Connection failed")

            client = ConcreteLLMClient.__new__(ConcreteLLMClient)
            client.config = config

            with pytest.raises(NetworkError) as exc_info:
                client._create_async_openai_client()

            assert "Не удалось создать AsyncOpenAI клиент" in str(
                exc_info.value)
            assert exc_info.value.get_context('endpoint') == config.endpoint

    def test_outlines_model_lazy_creation(self, config):
        """Тест ленивого создания Outlines модели."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            client = ConcreteLLMClient(config)

            # Мокаем outlines внутри метода
            with patch('outlines.from_openai') as mock_from_openai:
                mock_model = MagicMock()
                mock_from_openai.return_value = mock_model

                # Первый доступ - создание модели
                result1 = client.outlines_model
                assert result1 == mock_model
                mock_from_openai.assert_called_once_with(
                    client.openai_client,
                    config.model
                )

                # Второй доступ - возврат кэшированной модели
                result2 = client.outlines_model
                assert result2 == mock_model
                assert mock_from_openai.call_count == 1  # Не вызывается повторно

    def test_outlines_model_creation_error(self, config):
        """Тест ошибки создания Outlines модели."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            client = ConcreteLLMClient(config)

            with patch('outlines.from_openai') as mock_from_openai:
                mock_from_openai.side_effect = Exception(
                    "Model creation failed")

                with pytest.raises(NetworkError) as exc_info:
                    _ = client.outlines_model

                assert "Не удалось создать Outlines модель" in str(
                    exc_info.value)
                assert exc_info.value.get_context('model') == config.model

    def test_prepare_openai_params_defaults(self, config):
        """Тест подготовки параметров OpenAI с значениями по умолчанию."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            messages = [{"role": "user", "content": "test"}]
            params = client._prepare_openai_params(messages)

            assert params['messages'] == messages
            assert params['model'] == config.model
            assert params['temperature'] == config.temperature
            assert params['max_tokens'] == config.max_tokens
            assert params['top_p'] == config.top_p

    def test_prepare_openai_params_overrides(self, config):
        """Тест подготовки параметров OpenAI с переопределениями."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            messages = [{"role": "user", "content": "test"}]
            params = client._prepare_openai_params(
                messages,
                model="custom_model",
                temperature=0.9,
                max_tokens=2000,
                stream=True,
                custom_param="custom_value"
            )

            assert params['model'] == "custom_model"
            assert params['temperature'] == 0.9
            assert params['max_tokens'] == 2000
            assert params['stream'] is True
            assert params['custom_param'] == "custom_value"

    def test_messages_to_outlines_chat_success(self, config):
        """Тест успешной конвертации сообщений в Outlines Chat."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]

            with patch('outlines.inputs.Chat') as mock_chat:
                mock_chat_instance = MagicMock()
                mock_chat.return_value = mock_chat_instance

                result = client._messages_to_outlines_chat(messages)

                assert result == mock_chat_instance
                mock_chat.assert_called_once_with([
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ])

    def test_messages_to_outlines_chat_error(self, config):
        """Тест ошибки конвертации сообщений в Outlines Chat."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)
            messages = [{"role": "user", "content": "test"}]

            with patch('outlines.inputs.Chat') as mock_chat:
                mock_chat.side_effect = Exception("Chat creation failed")

                with pytest.raises(ValidationError) as exc_info:
                    client._messages_to_outlines_chat(messages)

                assert "Не удалось конвертировать сообщения в Outlines Chat" in str(
                    exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_openai_error_api_error(self, config):
        """Тест обработки API ошибки от OpenAI."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            # Создание mock ошибки с status_code
            error = Exception("API Error")
            error.status_code = 400
            error.response = {"error": "Bad request"}

            with pytest.raises(APIError) as exc_info:
                await client._handle_openai_error(error)

            assert exc_info.value.status_code == 400
            assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_openai_error_timeout(self, config):
        """Тест обработки ошибки таймаута."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            error = asyncio.TimeoutError("Request timeout")

            with pytest.raises(TimeoutError) as exc_info:
                await client._handle_openai_error(error)

            assert "Превышено время ожидания" in str(exc_info.value)
            assert exc_info.value.get_context(
                'timeout_config') == config.timeout_config

    @pytest.mark.asyncio
    async def test_handle_openai_error_network(self, config):
        """Тест обработки сетевой ошибки."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            error = httpx.ConnectError("Connection failed")

            with pytest.raises(NetworkError) as exc_info:
                await client._handle_openai_error(error)

            assert "Сетевая ошибка" in str(exc_info.value)
            assert exc_info.value.get_context('endpoint') == config.endpoint

    @pytest.mark.asyncio
    async def test_validate_pydantic_response_success(self, config):
        """Тест успешной валидации Pydantic ответа."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            response_json = '{"message": "test", "confidence": 0.9}'
            result = await client._validate_pydantic_response(response_json, ResponseModel)

            assert isinstance(result, ResponseModel)
            assert result.message == "test"
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_validate_pydantic_response_error(self, config):
        """Тест ошибки валидации Pydantic ответа."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            invalid_json = '{"invalid": "data"}'

            with pytest.raises(ValidationError) as exc_info:
                await client._validate_pydantic_response(invalid_json, ResponseModel)

            assert "Не удалось валидировать ответ по модели ResponseModel" in str(
                exc_info.value)
            assert exc_info.value.get_context('model') == "ResponseModel"

    @pytest.mark.asyncio
    async def test_close_client(self, config):
        """Тест закрытия клиента."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            client = ConcreteLLMClient(config)

            await client.close()

            mock_client.close.assert_called_once()
            assert client._outlines_model is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self, config):
        """Тест использования как async context manager."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            async with ConcreteLLMClient(config) as client:
                assert isinstance(client, ConcreteLLMClient)

            mock_client.close.assert_called_once()

    def test_repr(self, config):
        """Тест строкового представления клиента."""
        with patch('kraken_llm.client.base.AsyncOpenAI'):
            client = ConcreteLLMClient(config)

            repr_str = repr(client)

            assert "ConcreteLLMClient" in repr_str
            assert config.endpoint in repr_str
            assert config.model in repr_str
