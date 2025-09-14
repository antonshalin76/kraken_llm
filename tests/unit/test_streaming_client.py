"""
Unit тесты для StreamingLLMClient.

Этот модуль содержит тесты для проверки функциональности потокового LLM клиента,
включая обработку потоковых ответов, function/tool calling в потоке и валидацию параметров.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError


class TestStreamingLLMClient:
    """Тесты для StreamingLLMClient."""

    @pytest.fixture
    def config(self):
        """Фикстура конфигурации для тестов."""
        return LLMConfig(
            endpoint="http://test.example.com",
            api_key="test-key",
            model="test-model",
            temperature=0.7,
            max_tokens=1000
        )

    @pytest.fixture
    def client(self, config):
        """Фикстура клиента для тестов."""
        with patch('openai.AsyncOpenAI'):
            return StreamingLLMClient(config)

    def test_init(self, config):
        """Тест инициализации StreamingLLMClient."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            client = StreamingLLMClient(config)

            assert client.config == config
            assert client._function_registry == {}
            assert client._tool_registry == {}
            # Note: AsyncOpenAI is called in the base class constructor

    @pytest.mark.asyncio
    async def test_chat_completion_with_stream_false_raises_error(self, client):
        """Тест что chat_completion с stream=False выбрасывает ошибку."""
        messages = [{"role": "user", "content": "Привет"}]

        with pytest.raises(ValidationError) as exc_info:
            await client.chat_completion(messages, stream=False)

        assert "StreamingLLMClient предназначен для потоковых запросов" in str(
            exc_info.value)

    @pytest.mark.asyncio
    async def test_chat_completion_aggregates_stream(self, client):
        """Тест что chat_completion агрегирует потоковый ответ."""
        messages = [{"role": "user", "content": "Привет"}]

        # Мокаем chat_completion_stream с правильным async generator
        async def mock_stream(*args, **kwargs):
            yield "Привет"
            yield ", как"
            yield " дела?"

        # Заменяем метод на mock функцию
        client.chat_completion_stream = mock_stream

        result = await client.chat_completion(messages)

        assert result == "Привет, как дела?"

    def test_validate_stream_params_empty_messages(self, client):
        """Тест валидации пустого списка сообщений."""
        with pytest.raises(ValidationError) as exc_info:
            client._validate_stream_params([], None, None, None, None)

        assert "Список сообщений не может быть пустым" in str(exc_info.value)

    def test_validate_stream_params_function_call_without_functions(self, client):
        """Тест валидации function_call без functions."""
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ValidationError) as exc_info:
            client._validate_stream_params(messages, None, "auto", None, None)

        assert "function_call указан, но functions не предоставлены" in str(
            exc_info.value)

    def test_validate_stream_params_tool_choice_without_tools(self, client):
        """Тест валидации tool_choice без tools."""
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ValidationError) as exc_info:
            client._validate_stream_params(messages, None, None, None, "auto")

        assert "tool_choice указан, но tools не предоставлены" in str(
            exc_info.value)

    def test_validate_stream_params_functions_and_tools_together(self, client):
        """Тест валидации одновременного использования functions и tools."""
        messages = [{"role": "user", "content": "test"}]
        functions = [{"name": "test_func"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with pytest.raises(ValidationError) as exc_info:
            client._validate_stream_params(
                messages, functions, None, tools, None)

        assert "Нельзя использовать functions и tools одновременно" in str(
            exc_info.value)

    def test_validate_stream_params_invalid_message_structure(self, client):
        """Тест валидации некорректной структуры сообщений."""
        # Сообщение не является словарем
        with pytest.raises(ValidationError) as exc_info:
            client._validate_stream_params(["invalid"], None, None, None, None)

        assert "Сообщение 0 должно быть словарем" in str(exc_info.value)

        # Сообщение без role или content
        with pytest.raises(ValidationError) as exc_info:
            client._validate_stream_params(
                [{"role": "user"}], None, None, None, None)

        assert "должно содержать 'role' и 'content'" in str(exc_info.value)

        # Некорректная роль
        with pytest.raises(ValidationError) as exc_info:
            client._validate_stream_params(
                [{"role": "invalid", "content": "test"}], None, None, None, None)

        assert "Некорректная роль в сообщении 0: invalid" in str(
            exc_info.value)

    @pytest.mark.asyncio
    async def test_process_stream_chunk_content(self, client):
        """Тест обработки chunk с контентом."""
        # Создаем мок chunk с контентом
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "Тестовый контент"
        chunk.choices[0].delta.function_call = None
        chunk.choices[0].delta.tool_calls = None

        function_call_buffer = {"name": "", "arguments": ""}
        tool_calls_buffer = {}
        messages = [{"role": "user", "content": "test"}]

        result = await client._process_stream_chunk(
            chunk, function_call_buffer, tool_calls_buffer, messages
        )

        assert result == "Тестовый контент"

    @pytest.mark.asyncio
    async def test_process_stream_chunk_function_call(self, client):
        """Тест обработки chunk с function call."""
        # Создаем мок chunk с function call
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = None
        chunk.choices[0].delta.function_call = MagicMock()
        chunk.choices[0].delta.function_call.name = "test_function"
        chunk.choices[0].delta.function_call.arguments = '{"param": "value"}'
        chunk.choices[0].delta.tool_calls = None

        function_call_buffer = {"name": "", "arguments": ""}
        tool_calls_buffer = {}
        messages = [{"role": "user", "content": "test"}]

        result = await client._process_stream_chunk(
            chunk, function_call_buffer, tool_calls_buffer, messages
        )

        assert result is None  # Function call chunks не возвращают контент напрямую
        assert function_call_buffer["name"] == "test_function"
        assert function_call_buffer["arguments"] == '{"param": "value"}'

    @pytest.mark.asyncio
    async def test_process_stream_chunk_tool_calls(self, client):
        """Тест обработки chunk с tool calls."""
        # Создаем мок chunk с tool calls
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = None
        chunk.choices[0].delta.function_call = None

        # Мок tool call
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function = MagicMock()
        tool_call.function.name = "test_tool"
        tool_call.function.arguments = '{"param": "value"}'

        chunk.choices[0].delta.tool_calls = [tool_call]

        function_call_buffer = {"name": "", "arguments": ""}
        tool_calls_buffer = {}
        messages = [{"role": "user", "content": "test"}]

        result = await client._process_stream_chunk(
            chunk, function_call_buffer, tool_calls_buffer, messages
        )

        assert result is None  # Tool call chunks не возвращают контент напрямую
        assert "call_123" in tool_calls_buffer
        assert tool_calls_buffer["call_123"]["function"]["name"] == "test_tool"
        assert tool_calls_buffer["call_123"]["function"][
            "arguments"] == '{"param": "value"}'

    @pytest.mark.asyncio
    async def test_execute_aggregated_function_call_success(self, client):
        """Тест успешного выполнения агрегированного function call."""
        # Регистрируем тестовую функцию
        def test_function(param):
            return f"Результат: {param}"

        client.register_function("test_function", test_function)

        function_call_data = {
            "name": "test_function",
            "arguments": '{"param": "тест"}'
        }
        messages = [{"role": "user", "content": "test"}]

        result = await client._execute_aggregated_function_call(function_call_data, messages)

        assert "Результат функции test_function: Результат: тест" in result

    @pytest.mark.asyncio
    async def test_execute_aggregated_function_call_not_registered(self, client):
        """Тест выполнения незарегистрированной функции."""
        function_call_data = {
            "name": "unknown_function",
            "arguments": '{"param": "тест"}'
        }
        messages = [{"role": "user", "content": "test"}]

        result = await client._execute_aggregated_function_call(function_call_data, messages)

        assert "функция unknown_function не найдена" in result

    @pytest.mark.asyncio
    async def test_execute_aggregated_function_call_invalid_json(self, client):
        """Тест выполнения function call с некорректным JSON."""
        function_call_data = {
            "name": "test_function",
            "arguments": 'invalid json'
        }
        messages = [{"role": "user", "content": "test"}]

        result = await client._execute_aggregated_function_call(function_call_data, messages)

        assert "некорректные аргументы функции" in result

    @pytest.mark.asyncio
    async def test_execute_aggregated_tool_calls_success(self, client):
        """Тест успешного выполнения агрегированных tool calls."""
        # Регистрируем тестовый инструмент
        def test_tool(param):
            return f"Результат инструмента: {param}"

        client.register_tool("test_tool", test_tool)

        tool_calls_data = {
            "call_123": {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param": "тест"}'
                }
            }
        }
        messages = [{"role": "user", "content": "test"}]

        result = await client._execute_aggregated_tool_calls(tool_calls_data, messages)

        assert "Tool test_tool: Результат инструмента: тест" in result

    @pytest.mark.asyncio
    async def test_execute_function_sync(self, client):
        """Тест выполнения синхронной функции."""
        def sync_function(x, y):
            return x + y

        result = await client._execute_function(sync_function, {"x": 5, "y": 3})

        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_function_async(self, client):
        """Тест выполнения асинхронной функции."""
        async def async_function(x, y):
            await asyncio.sleep(0.01)  # Имитация асинхронной работы
            return x * y

        result = await client._execute_function(async_function, {"x": 4, "y": 3})

        assert result == 12

    def test_register_function(self, client):
        """Тест регистрации функции."""
        def test_function():
            return "test"

        client.register_function("test_func", test_function, "Test function")

        assert "test_func" in client._function_registry
        assert client._function_registry["test_func"] == test_function
        assert "test_func" in client.get_registered_functions()

    def test_register_tool(self, client):
        """Тест регистрации инструмента."""
        def test_tool():
            return "test"

        client.register_tool("test_tool", test_tool, "Test tool")

        assert "test_tool" in client._tool_registry
        assert client._tool_registry["test_tool"] == test_tool
        assert "test_tool" in client.get_registered_tools()

    @pytest.mark.asyncio
    async def test_chat_completion_structured_not_supported(self, client):
        """Тест что structured output не поддерживается."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            text: str

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(NotImplementedError) as exc_info:
            await client.chat_completion_structured(messages, TestModel)

        assert "StreamingLLMClient не поддерживает structured output" in str(
            exc_info.value)


class TestStreamingClientIntegration:
    """Интеграционные тесты для StreamingLLMClient."""

    @pytest.fixture
    def config(self):
        """Фикстура конфигурации для интеграционных тестов."""
        return LLMConfig(
            endpoint="http://test.example.com",
            api_key="test-key",
            model="test-model"
        )

    @pytest.mark.asyncio
    async def test_full_streaming_flow(self, config):
        """Тест полного потокового процесса."""
        with patch('openai.AsyncOpenAI') as mock_openai_class:
            # Настраиваем мок для AsyncOpenAI
            mock_openai = AsyncMock()
            mock_openai_class.return_value = mock_openai

            # Создаем функцию для создания мок потока
            def create_mock_stream():
                async def mock_stream():
                    # Chunk с контентом
                    chunk1 = MagicMock()
                    chunk1.choices = [MagicMock()]
                    chunk1.choices[0].delta = MagicMock()
                    chunk1.choices[0].delta.content = "Привет"
                    chunk1.choices[0].delta.function_call = None
                    chunk1.choices[0].delta.tool_calls = None
                    yield chunk1

                    # Еще один chunk с контентом
                    chunk2 = MagicMock()
                    chunk2.choices = [MagicMock()]
                    chunk2.choices[0].delta = MagicMock()
                    chunk2.choices[0].delta.content = ", мир!"
                    chunk2.choices[0].delta.function_call = None
                    chunk2.choices[0].delta.tool_calls = None
                    yield chunk2
                return mock_stream()

            # Настраиваем мок для создания нового потока при каждом вызове
            mock_openai.chat.completions.create.side_effect = lambda **kwargs: create_mock_stream()

            # Создаем клиент и выполняем запрос
            client = StreamingLLMClient(config)

            # Заменяем openai_client на наш мок после создания клиента
            client.openai_client = mock_openai

            messages = [{"role": "user", "content": "Скажи привет"}]

            # Тестируем потоковый метод
            chunks = []
            async for chunk in client.chat_completion_stream(messages):
                chunks.append(chunk)

            assert chunks == ["Привет", ", мир!"]

            # Тестируем агрегированный метод
            result = await client.chat_completion(messages)
            assert result == "Привет, мир!"

            # Проверяем что API был вызван с правильными параметрами
            mock_openai.chat.completions.create.assert_called()
            call_args = mock_openai.chat.completions.create.call_args[1]
            assert call_args["messages"] == messages
            assert call_args["stream"] is True
            assert call_args["model"] == "test-model"
