"""
Unit тесты для StructuredLLMClient.

Тестирует функциональность structured output клиента включая валидацию,
интеграцию с Outlines, обработку ошибок и различные режимы работы.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

from pydantic import BaseModel, Field

from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError
from kraken_llm.exceptions.api import APIError


# Тестовые Pydantic модели
class SimpleModel(BaseModel):
    """Простая модель для тестирования."""
    name: str = Field(..., description="Имя пользователя")
    age: int = Field(..., ge=0, le=150, description="Возраст пользователя")
    email: Optional[str] = Field(None, description="Email адрес")


class ComplexModel(BaseModel):
    """Сложная модель с вложенными объектами."""
    user: SimpleModel
    tags: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    is_active: bool = True


class NestedModel(BaseModel):
    """Модель с глубокой вложенностью."""
    level1: dict
    level2: Optional[dict] = None


@pytest.fixture
def mock_config():
    """Мок конфигурации для тестов."""
    return LLMConfig(
        endpoint="http://test-endpoint:8080",
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        max_tokens=1000,
    )


@pytest.fixture
async def structured_client(mock_config):
    """Фикстура StructuredLLMClient с моками."""
    # Мокаем только базовый класс
    with patch.object(StructuredLLMClient, '_create_async_openai_client') as mock_create_openai, \
            patch.object(StructuredLLMClient, '_validate_config') as mock_validate:

        # Настройка моков
        mock_openai_client = AsyncMock()
        mock_openai_client.close = AsyncMock()
        mock_create_openai.return_value = mock_openai_client

        # Создаем клиент
        client = StructuredLLMClient(mock_config)

        yield client

        await client.close()


class TestStructuredLLMClientInit:
    """Тесты инициализации StructuredLLMClient."""

    def test_init_success(self, mock_config):
        """Тест успешной инициализации клиента."""
        with patch.object(StructuredLLMClient, '_create_async_openai_client') as mock_create_openai, \
                patch.object(StructuredLLMClient, '_validate_config') as mock_validate:

            mock_create_openai.return_value = AsyncMock()

            client = StructuredLLMClient(mock_config)

            assert client.config == mock_config
            assert client._outlines_model is None  # Ленивая инициализация
            mock_create_openai.assert_called_once()

    def test_repr(self, mock_config):
        """Тест строкового представления клиента."""
        with patch.object(StructuredLLMClient, '_create_async_openai_client') as mock_create_openai, \
                patch.object(StructuredLLMClient, '_validate_config') as mock_validate:

            mock_create_openai.return_value = AsyncMock()

            client = StructuredLLMClient(mock_config)
            repr_str = repr(client)

            assert "StructuredLLMClient" in repr_str
            assert mock_config.endpoint in repr_str
            assert mock_config.model in repr_str


class TestStructuredLLMClientValidation:
    """Тесты валидации параметров."""

    async def test_validate_empty_messages(self, structured_client):
        """Тест валидации с пустым списком сообщений."""
        with pytest.raises(ValidationError, match="не может быть пустым"):
            await structured_client.chat_completion([], response_model=SimpleModel)

    async def test_validate_none_model(self, structured_client):
        """Тест валидации с None моделью."""
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(ValidationError, match="response_model обязателен"):
            await structured_client.chat_completion(messages, response_model=None)

    async def test_validate_invalid_model_type(self, structured_client):
        """Тест валидации с некорректным типом модели."""
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(ValidationError, match="должен быть Pydantic BaseModel"):
            await structured_client.chat_completion(messages, response_model=str)


class TestStructuredLLMClientNonStream:
    """Тесты non-streaming structured output."""

    async def test_structured_non_stream_success(self, structured_client):
        """Тест успешного non-streaming structured output."""
        messages = [
            {"role": "user", "content": "Generate user data"}
        ]

        # Мокаем валидатор
        expected_result = SimpleModel(
            name="John Doe", age=30, email="john@example.com")

        with patch.object(structured_client.validator, 'validate_response') as mock_validate:
            mock_validate.return_value = expected_result

            # Мокаем _server_supports_non_streaming чтобы использовать OpenAI API
            with patch.object(structured_client, '_server_supports_non_streaming', return_value=True):
                # Мокаем OpenAI API response
                mock_response = MagicMock()
                mock_response.id = "test-response-id"
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'

                structured_client.openai_client.chat.completions.create = AsyncMock(
                    return_value=mock_response)

                result = await structured_client.chat_completion(
                    messages, response_model=SimpleModel, stream=False
                )

                assert isinstance(result, SimpleModel)
                assert result.name == "John Doe"
                assert result.age == 30
                assert result.email == "john@example.com"


class TestStructuredLLMClientStream:
    """Тесты streaming structured output."""

    async def test_structured_stream_success(self, structured_client):
        """Тест успешного streaming structured output."""
        messages = [
            {"role": "user", "content": "Generate user data"}
        ]

        # Мокаем валидатор
        expected_result = SimpleModel(name="Alice", age=28)

        with patch.object(structured_client.validator, 'validate_response') as mock_validate:
            mock_validate.return_value = expected_result

            # Создаем мок для streaming chunks
            async def mock_stream():
                chunks = [
                    MagicMock(
                        choices=[MagicMock(delta=MagicMock(content='{"name":'))]),
                    MagicMock(
                        choices=[MagicMock(delta=MagicMock(content=' "Alice",'))]),
                    MagicMock(
                        choices=[MagicMock(delta=MagicMock(content=' "age": 28}'))]),
                ]
                for chunk in chunks:
                    yield chunk

            structured_client.openai_client.chat.completions.create = AsyncMock(
                return_value=mock_stream())

            result = await structured_client.chat_completion(
                messages, response_model=SimpleModel, stream=True
            )

            assert isinstance(result, SimpleModel)
            assert result.name == "Alice"
            assert result.age == 28


class TestStructuredLLMClientPublicAPI:
    """Тесты публичного API."""

    async def test_chat_completion_structured_non_stream(self, structured_client):
        """Тест публичного метода без streaming."""
        messages = [
            {"role": "user", "content": "Generate user data"}
        ]

        expected_result = SimpleModel(name="API Test", age=40)

        with patch.object(structured_client, '_structured_non_stream') as mock_non_stream:
            mock_non_stream.return_value = expected_result

            result = await structured_client.chat_completion_structured(
                messages, SimpleModel, stream=False
            )

            assert result == expected_result
            mock_non_stream.assert_called_once()

    async def test_chat_completion_structured_stream(self, structured_client):
        """Тест публичного метода со streaming."""
        messages = [
            {"role": "user", "content": "Generate user data"}
        ]

        expected_result = SimpleModel(name="Stream Test", age=35)

        with patch.object(structured_client, '_structured_stream') as mock_stream:
            mock_stream.return_value = expected_result

            result = await structured_client.chat_completion_structured(
                messages, SimpleModel, stream=True
            )

            assert result == expected_result
            mock_stream.assert_called_once()

    async def test_chat_completion_structured_default_non_stream(self, structured_client):
        """Тест публичного метода с default режимом (non-stream)."""
        messages = [
            {"role": "user", "content": "Generate user data"}
        ]

        expected_result = SimpleModel(name="Default Test", age=30)

        with patch.object(structured_client, '_structured_non_stream') as mock_non_stream:
            mock_non_stream.return_value = expected_result

            result = await structured_client.chat_completion_structured(
                messages, SimpleModel  # stream не указан
            )

            assert result == expected_result
            mock_non_stream.assert_called_once()

    async def test_chat_completion_not_supported(self, structured_client):
        """Тест что обычный chat_completion требует response_model."""
        messages = [
            {"role": "user", "content": "Test"}
        ]

        with pytest.raises(ValidationError, match="response_model обязателен"):
            await structured_client.chat_completion(messages)

    async def test_chat_completion_stream_not_supported(self, structured_client):
        """Тест что обычный streaming не поддерживается."""
        messages = [
            {"role": "user", "content": "Test"}
        ]

        # Метод должен выбросить NotImplementedError при вызове
        with pytest.raises(NotImplementedError, match="не поддерживает обычный streaming"):
            # Поскольку метод возвращает coroutine, а не async generator, просто await его
            await structured_client.chat_completion_stream(messages)


class TestStructuredLLMClientStreamingChunks:
    """Тесты потокового API с chunks."""

    async def test_structured_completion_method(self, structured_client):
        """Тест метода structured_completion."""
        messages = [
            {"role": "user", "content": "Generate user data"}
        ]

        expected_result = SimpleModel(name="Structured Test", age=25)

        with patch.object(structured_client, 'chat_completion') as mock_chat:
            mock_chat.return_value = expected_result

            result = await structured_client.structured_completion(
                messages, SimpleModel, stream=False
            )

            assert result == expected_result
            mock_chat.assert_called_once_with(
                messages=messages,
                model=None,
                temperature=None,
                max_tokens=None,
                stream=False,
                response_model=SimpleModel
            )


class TestStructuredLLMClientUtilities:
    """Тесты утилитарных методов."""

    def test_enhance_messages_for_json(self, structured_client):
        """Тест улучшения сообщений для JSON генерации."""
        messages = [
            {"role": "user", "content": "Generate user data"}
        ]

        enhanced = structured_client._enhance_messages_for_json(
            messages, SimpleModel)

        assert len(enhanced) >= len(messages)
        # Проверяем, что добавлено системное сообщение
        assert any(msg["role"] == "system" for msg in enhanced)

        # Проверяем, что в системном сообщении есть инструкции по JSON
        system_msg = next(msg for msg in enhanced if msg["role"] == "system")
        assert "JSON" in system_msg["content"]
        # Проверяем наличие схемы (может быть на русском или английском)
        content_lower = system_msg["content"].lower()
        assert "схема" in content_lower or "schema" in content_lower

    def test_generate_example_json(self, structured_client):
        """Тест генерации примера JSON."""
        schema = SimpleModel.model_json_schema()
        example = structured_client._generate_example_json(schema)

        assert isinstance(example, dict)
        assert "name" in example
        assert "age" in example
        assert isinstance(example["name"], str)
        assert isinstance(example["age"], int)

    def test_extract_json_from_response(self, structured_client):
        """Тест извлечения JSON из ответа."""
        # Тест с чистым JSON
        json_response = '{"name": "John", "age": 30}'
        extracted = structured_client._extract_json_from_response(
            json_response)
        assert extracted == json_response

        # Тест с JSON в code block
        markdown_response = '```json\n{"name": "John", "age": 30}\n```'
        extracted = structured_client._extract_json_from_response(
            markdown_response)
        assert extracted == '{"name": "John", "age": 30}'

        # Тест с дополнительным текстом
        mixed_response = 'Here is the data: {"name": "John", "age": 30} and some more text'
        extracted = structured_client._extract_json_from_response(
            mixed_response)
        assert extracted == '{"name": "John", "age": 30}'


class TestStructuredLLMClientComplexModels:
    """Тесты с сложными Pydantic моделями."""

    async def test_complex_model_success(self, structured_client):
        """Тест с комплексной моделью."""
        messages = [
            {"role": "user", "content": "Generate complex user data"}
        ]

        # Создаем сложный результат
        user_data = SimpleModel(name="Complex User",
                                age=30, email="complex@example.com")
        expected_result = ComplexModel(
            user=user_data,
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
            is_active=True
        )

        with patch.object(structured_client.validator, 'validate_response') as mock_validate:
            mock_validate.return_value = expected_result

            # Мокаем _server_supports_non_streaming чтобы использовать OpenAI API
            with patch.object(structured_client, '_server_supports_non_streaming', return_value=True):
                # Мокаем OpenAI API response
                mock_response = MagicMock()
                mock_response.id = "test-complex-response-id"
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = '{"user": {"name": "Complex User", "age": 30, "email": "complex@example.com"}, "tags": ["tag1", "tag2"], "metadata": {"key": "value"}, "is_active": true}'

                structured_client.openai_client.chat.completions.create = AsyncMock(
                    return_value=mock_response)

                result = await structured_client.chat_completion(
                    messages, response_model=ComplexModel, stream=False
                )

                assert isinstance(result, ComplexModel)
                assert isinstance(result.user, SimpleModel)
                assert result.user.name == "Complex User"
                assert result.tags == ["tag1", "tag2"]
                assert result.metadata == {"key": "value"}
                assert result.is_active is True


class TestStructuredLLMClientErrorHandling:
    """Тесты обработки ошибок."""

    async def test_validation_error_propagation(self, structured_client):
        """Тест пробрасывания ValidationError."""
        messages = []  # Пустой список должен вызвать ValidationError

        with pytest.raises(ValidationError, match="не может быть пустым"):
            await structured_client.chat_completion_structured(messages, SimpleModel)

    async def test_api_error_handling(self, structured_client):
        """Тест обработки API ошибок."""
        messages = [
            {"role": "user", "content": "Test"}
        ]

        # Мокаем ошибку в OpenAI API
        structured_client.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error"))

        with patch.object(structured_client, '_server_supports_non_streaming', return_value=True):
            with pytest.raises(Exception):  # Ошибка должна пробрасываться
                await structured_client.chat_completion_structured(messages, SimpleModel)


@pytest.mark.asyncio
class TestStructuredLLMClientAsyncContext:
    """Тесты async context manager."""

    async def test_async_context_manager(self, mock_config):
        """Тест использования как async context manager."""
        with patch.object(StructuredLLMClient, '_create_async_openai_client') as mock_create_openai, \
                patch.object(StructuredLLMClient, '_validate_config') as mock_validate:

            mock_openai_instance = AsyncMock()
            mock_create_openai.return_value = mock_openai_instance

            async with StructuredLLMClient(mock_config) as client:
                assert isinstance(client, StructuredLLMClient)

            # Проверяем, что close был вызван
            mock_openai_instance.close.assert_called_once()

    async def test_close_method(self, structured_client):
        """Тест метода close."""
        await structured_client.close()

        # Проверяем, что AsyncOpenAI клиент был закрыт
        structured_client.openai_client.close.assert_called_once()

        # Проверяем, что Outlines модель была сброшена
        assert structured_client._outlines_model is None
