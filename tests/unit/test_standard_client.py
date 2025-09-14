"""
Модульные тесты для StandardLLMClient.

Эти тесты проверяют функциональность StandardLLMClient в изоляции,
фокусируясь на валидации параметров, поведении методов и обработке ошибок.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from kraken_llm.client.standard import StandardLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError


class TestStandardLLMClient:
    """Модульные тесты для StandardLLMClient."""

    @pytest.fixture
    def config(self):
        """Тестовая конфигурация."""
        return LLMConfig(
            endpoint="http://test.local",
            api_key="test_key",
            model="test_model",
            temperature=0.5,
            max_tokens=200,
        )

    @pytest.fixture
    def mock_openai_client(self):
        """Мок AsyncOpenAI клиента."""
        return AsyncMock()

    @pytest.fixture
    def client(self, config):
        """Экземпляр StandardLLMClient с замоканными зависимостями."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()
            client = StandardLLMClient(config)
            return client

    def test_initialization(self, config):
        """Тест инициализации клиента."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()

            client = StandardLLMClient(config)

            assert client.config == config
            assert client._function_registry == {}
            assert client._tool_registry == {}
            mock_openai.assert_called_once()

    def test_register_function(self, client):
        """Тест регистрации функции."""
        def test_func(x: int) -> int:
            return x * 2

        client.register_function(
            name="double",
            function=test_func,
            description="Удваивает число"
        )

        assert "double" in client._function_registry
        assert client._function_registry["double"] == test_func
        assert client.get_registered_functions() == ["double"]

    def test_register_tool(self, client):
        """Тест регистрации инструмента."""
        def test_tool(x: str) -> str:
            return x.upper()

        client.register_tool(
            name="uppercase",
            tool=test_tool,
            description="Преобразует в верхний регистр"
        )

        assert "uppercase" in client._tool_registry
        assert client._tool_registry["uppercase"] == test_tool
        assert client.get_registered_tools() == ["uppercase"]

    def test_multiple_registrations(self, client):
        """Тест множественной регистрации функций и инструментов."""
        def func1():
            return "func1"

        def func2():
            return "func2"

        def tool1():
            return "tool1"

        def tool2():
            return "tool2"

        client.register_function("func1", func1)
        client.register_function("func2", func2)
        client.register_tool("tool1", tool1)
        client.register_tool("tool2", tool2)

        assert set(client.get_registered_functions()) == {"func1", "func2"}
        assert set(client.get_registered_tools()) == {"tool1", "tool2"}

    def test_validate_chat_completion_params_empty_messages(self, client):
        """Тест валидации с пустыми сообщениями."""
        with pytest.raises(ValidationError, match="Список сообщений не может быть пустым"):
            client._validate_chat_completion_params(
                messages=[],
                stream=None,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None
            )

    def test_validate_chat_completion_params_streaming_not_supported(self, client):
        """Тест валидации отклоняет стриминг."""
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ValidationError, match="не поддерживает streaming"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=True,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None
            )

    def test_validate_chat_completion_params_function_call_without_functions(self, client):
        """Тест валидации отклоняет function_call без functions."""
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ValidationError, match="function_call указан, но functions не предоставлены"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=None,
                function_call="auto",
                tools=None,
                tool_choice=None
            )

    def test_validate_chat_completion_params_tool_choice_without_tools(self, client):
        """Тест валидации отклоняет tool_choice без tools."""
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ValidationError, match="tool_choice указан, но tools не предоставлены"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice="auto"
            )

    def test_validate_chat_completion_params_functions_and_tools_together(self, client):
        """Тест валидации отклоняет одновременное использование functions и tools."""
        messages = [{"role": "user", "content": "test"}]
        functions = [{"name": "test_func"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with pytest.raises(ValidationError, match="Нельзя использовать functions и tools одновременно"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=functions,
                function_call=None,
                tools=tools,
                tool_choice=None
            )

    def test_validate_chat_completion_params_invalid_message_structure(self, client):
        """Тест валидации структуры сообщений."""
        # Отсутствует content
        messages = [{"role": "user"}]

        with pytest.raises(ValidationError, match="должно содержать 'role' и 'content'"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None
            )

        # Отсутствует role
        messages = [{"content": "test"}]

        with pytest.raises(ValidationError, match="должно содержать 'role' и 'content'"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None
            )

        # Сообщение не является словарем
        messages = ["invalid"]

        with pytest.raises(ValidationError, match="должно быть словарем"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None
            )

    def test_validate_chat_completion_params_invalid_role(self, client):
        """Тест валидации ролей сообщений."""
        messages = [{"role": "invalid_role", "content": "test"}]

        with pytest.raises(ValidationError, match="Некорректная роль"):
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None
            )

    def test_validate_chat_completion_params_valid_roles(self, client):
        """Тест валидации принимает корректные роли."""
        valid_roles = ["system", "user", "assistant", "function", "tool"]

        for role in valid_roles:
            messages = [{"role": role, "content": "test"}]

            # Не должно вызывать исключений
            client._validate_chat_completion_params(
                messages=messages,
                stream=None,
                functions=None,
                function_call=None,
                tools=None,
                tool_choice=None
            )

    def test_validate_chat_completion_params_valid_combinations(self, client):
        """Тест валидации принимает корректные комбинации параметров."""
        messages = [{"role": "user", "content": "test"}]

        # Functions с function_call
        functions = [{"name": "test_func"}]
        client._validate_chat_completion_params(
            messages=messages,
            stream=None,
            functions=functions,
            function_call="auto",
            tools=None,
            tool_choice=None
        )

        # Tools с tool_choice
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        client._validate_chat_completion_params(
            messages=messages,
            stream=None,
            functions=None,
            function_call=None,
            tools=tools,
            tool_choice="auto"
        )

        # Без functions или tools
        client._validate_chat_completion_params(
            messages=messages,
            stream=None,
            functions=None,
            function_call=None,
            tools=None,
            tool_choice=None
        )

    @pytest.mark.asyncio
    async def test_execute_function_sync(self, client):
        """Тест выполнения синхронной функции."""
        def sync_func(x: int, y: int) -> int:
            return x + y

        result = await client._execute_function(sync_func, {"x": 5, "y": 3})
        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_function_async(self, client):
        """Тест выполнения асинхронной функции."""
        async def async_func(x: int, y: int) -> int:
            return x * y

        result = await client._execute_function(async_func, {"x": 4, "y": 6})
        assert result == 24

    @pytest.mark.asyncio
    async def test_execute_tool(self, client):
        """Тест выполнения инструмента (аналогично функции)."""
        def tool_func(text: str) -> str:
            return text.upper()

        result = await client._execute_tool(tool_func, {"text": "hello"})
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_fallback_streaming_method(self, client):
        """Тест что метод стриминга предоставляет резервную функциональность."""
        messages = [{"role": "user", "content": "test"}]

        # Мокаем метод chat_completion чтобы избежать сетевых вызовов
        with patch.object(client, 'chat_completion', return_value="test response") as mock_chat:
            chunks = []
            async for chunk in client.chat_completion_stream(messages=messages):
                chunks.append(chunk)

            # Должен был вызвать chat_completion один раз
            mock_chat.assert_called_once_with(messages)

            # Должен вернуть хотя бы один чанк
            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_unsupported_structured_method(self, client):
        """Тест что метод структурированного вывода вызывает NotImplementedError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            text: str

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(NotImplementedError, match="не поддерживает structured output"):
            await client.chat_completion_structured(
                messages=messages,
                response_model=TestModel
            )

    def test_client_repr(self, client):
        """Тест строкового представления клиента."""
        repr_str = repr(client)
        assert "StandardLLMClient" in repr_str
        assert "http://test.local" in repr_str
        assert "test_model" in repr_str

    @pytest.mark.asyncio
    async def test_client_context_manager(self, config):
        """Тест клиента как асинхронного контекстного менеджера."""
        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()

            async with StandardLLMClient(config) as client:
                assert isinstance(client, StandardLLMClient)
                # Клиент должен быть доступен в контексте
                assert client.config == config

            # close() должен был быть вызван
            client.openai_client.close.assert_called_once()

    def test_registry_isolation(self):
        """Тест что разные экземпляры клиента имеют изолированные реестры."""
        config = LLMConfig(api_key="test")

        with patch('kraken_llm.client.base.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = AsyncMock()

            client1 = StandardLLMClient(config)
            client2 = StandardLLMClient(config)

            # Регистрируем разные функции в каждом клиенте
            client1.register_function("func1", lambda: "client1")
            client2.register_function("func2", lambda: "client2")

            # Реестры должны быть изолированы
            assert client1.get_registered_functions() == ["func1"]
            assert client2.get_registered_functions() == ["func2"]

            assert "func1" not in client2._function_registry
            assert "func2" not in client1._function_registry

    def test_function_overwrite(self, client):
        """Тест что регистрация функции с тем же именем перезаписывает её."""
        def func1():
            return "first"

        def func2():
            return "second"

        client.register_function("test_func", func1)
        assert client._function_registry["test_func"] == func1

        client.register_function("test_func", func2)
        assert client._function_registry["test_func"] == func2

        # Должна остаться только одна зарегистрированная функция
        assert len(client.get_registered_functions()) == 1

    def test_tool_overwrite(self, client):
        """Тест что регистрация инструмента с тем же именем перезаписывает его."""
        def tool1():
            return "first"

        def tool2():
            return "second"

        client.register_tool("test_tool", tool1)
        assert client._tool_registry["test_tool"] == tool1

        client.register_tool("test_tool", tool2)
        assert client._tool_registry["test_tool"] == tool2

        # Должен остаться только один зарегистрированный инструмент
        assert len(client.get_registered_tools()) == 1

    def test_empty_registries_initially(self, client):
        """Тест что реестры изначально пусты."""
        assert client.get_registered_functions() == []
        assert client.get_registered_tools() == []
        assert len(client._function_registry) == 0
        assert len(client._tool_registry) == 0
